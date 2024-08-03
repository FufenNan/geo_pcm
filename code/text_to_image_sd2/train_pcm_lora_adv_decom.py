#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import sys
import gc
import math
import random
import shutil
import logging
import argparse
import functools
from pathlib import Path
from packaging import version

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim import RMSprop

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState

import transformers
from transformers import (
    AutoTokenizer, CLIPTextModel, PretrainedConfig, 
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    CLIPTokenizer
)
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    DiffusionPipeline, DDPMScheduler, DDIMScheduler, AutoencoderKL
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version, deprecate, is_wandb_available, 
    make_image_grid, is_xformers_available
)
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from huggingface_hub import create_repo
from tqdm.auto import tqdm
from models.unet_2d_condition import UNet2DConditionModel
from utils.de_normalized import align_scale_shift
from utils.depth2normal import *
from utils.dataset_configuration import (
    prepare_dataset, depth_scale_shift_normalization, 
    resize_max_res_tensor
)


from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from scheduling_ddpm_modified import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from discriminator_sd2 import Discriminator


MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)


def get_module_kohya_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(
                module.peft_config[adapter_name].lora_alpha
            ).to(dtype)

    return kohya_ss_state_dict


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, sample_size):
        """
        Args:
            img_dir (string): Directory with all the images and text files.
            sample_size (tuple): Desired sample size as (height, width).
        """
        self.img_dir = img_dir
        self.sample_size = sample_size
        self.img_names = [
            f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.sample_size, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        while True:
            img_name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            text_name = img_name.rsplit(".", 1)[0] + ".txt"
            text_path = os.path.join(self.img_dir, text_name)
            try:
                with open(text_path, "r") as f:
                    text = f.read().strip()
            except FileNotFoundError:

                continue

            return image, text


def log_validation(
    vae, unet, args, accelerator, weight_dtype, step, cfg, num_inference_step
):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_teacher_model,
        vae=vae,
        scheduler=DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
            clip_sample = False, # important. DDIM will apply True as default which causes inference degradation.
            set_alpha_to_one = False,
        ),  # DDIM should just work well. See our discussion on parameterization in the paper.
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.set_progress_bar_config(disable=True)
    lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
    pipeline.load_lora_weights(lora_state_dict)
    pipeline.fuse_lora()
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        with torch.autocast("cuda", dtype=weight_dtype):
            images = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_step,
                num_images_per_prompt=4,
                generator=generator,
                guidance_scale=cfg,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    f"{cfg}: " + validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({f"validation-{cfg}": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions_target(index, selected_indices):
    c_skip = torch.isin(index, selected_indices).float()
    c_out = 1.0 - c_skip
    return c_skip, c_out


def scalings_for_boundary_conditions_online(index, selected_indices):
    c_skip = torch.zeros_like(index).float()
    c_out = torch.ones_like(index).float()
    return c_skip, c_out


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")
    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        self.step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_timesteps_prev = np.asarray([0] + self.ddim_timesteps[:-1].tolist())
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_timesteps_prev = torch.from_numpy(self.ddim_timesteps_prev).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_timesteps_prev = self.ddim_timesteps_prev.to(device)

        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

    def ddim_style_multiphase_pred(
        self, pred_x0, pred_noise, timestep_index, multiphase
    ):
        inference_indices = np.linspace(
            0, len(self.ddim_timesteps), num=multiphase, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            torch.from_numpy(inference_indices).long().to(self.ddim_timesteps.device)
        )
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0)
        )
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index = inference_indices[last_valid_index]
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, self.ddim_timesteps_prev[timestep_index]


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        use_auth_token=True,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def pyramid_noise_like(x, timesteps, discount=0.9):
    b, c, w_ori, h_ori = x.shape 
    u = nn.Upsample(size=(w_ori, h_ori), mode='bilinear')
    noise = torch.randn_like(x)
    scale = 1.5
    for i in range(10):
        r = np.random.random()*scale + scale # Rather than always going 2x, 
        w, h = max(1, int(w_ori/(r**i))), max(1, int(h_ori/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * (timesteps[...,None,None,None]/1000) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/haoyum3/GeoWizard_edit/hypersim",
        help="Path to the training dataloader.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA projection matrix.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--not_apply_cfg_solver", action="store_true")
    parser.add_argument("--multiphase", default=8, type=int)
    parser.add_argument("--adv_weight", default=0.1, type=float)
    parser.add_argument("--adv_lr", default=1e-5, type=float)


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")
    return args


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        # this should be with the device id?
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="scheduler",
        revision=args.teacher_revision,
    )

    # The scheduler calculates the alpha and sigma schedule for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizers from SD-XL checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )

    # 3. Load text encoders from SD-1.5 checkpoint.
    # import correct text encoder classes
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.teacher_revision,
    )

    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD-XL checkpoint
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', subfolder="image_encoder",revision=args.teacher_revision
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', subfolder="feature_extractor",revision=args.teacher_revision
    )
    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)
    
    discriminator = Discriminator(teacher_unet)
    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    discriminator.unet.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    discriminator_params = []
    for param in discriminator.heads.parameters():
        param.requires_grad = True
        discriminator_params.append(param)

    # 7. Create online (`unet`) student U-Nets.
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
    )

    unet.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ],
    )
    unet = get_peft_model(unet, lora_config)

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                lora_state_dict = get_peft_model_state_dict(
                    unet_, adapter_name="default"
                )
                StableDiffusionPipeline.save_lora_weights(
                    os.path.join(output_dir, "unet_lora"), lora_state_dict
                )
                # save weights in peft format to be able to load them back
                unet_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            unet_ = accelerator.unwrap_model(unet)
            unet_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        teacher_unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_discriminator = optimizer_class(
        discriminator_params,
        lr=args.adv_lr,
        betas=(0, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train
        )
        return {"prompt_embeds": prompt_embeds}

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=args.proportion_empty_prompts,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    with accelerator.main_process_first():
        train_dataloader, dataset_config_dict = prepare_dataset(data_dir=args.dataset_path,
                                                                    batch_size=args.train_batch_size,
                                                                    test_batch=1,
                                                                    datathread=args.dataloader_num_workers,
                                                                    logger=logger)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        unet,
        discriminator,
        optimizer,
        optimizer_discriminator,
        lr_scheduler,
        train_dataloader,
    ) = accelerator.prepare(
        unet,
        discriminator,
        optimizer,
        optimizer_discriminator,
        lr_scheduler,
        train_dataloader,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    #[bsz,77]
    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
    ).input_ids.to(accelerator.device)

    #[bsz,77,1024]
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                image_data = batch['rgb'].clip(-1., 1.)
                image_data.to(accelerator.device, non_blocking=True)
                image_data_resized = resize_max_res_tensor(image_data, mode='rgb')

                device = image_data.device
                
                imgs_in_proc = TF.resize((image_data_resized +1)/2, 
                    (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), 
                    interpolation=InterpolationMode.BICUBIC, 
                    antialias=True
                )
                # do the normalization in float32 to preserve precision
                imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std).to(weight_dtype)        
                imgs_embed= image_encoder(imgs_in_proc).image_embeds.unsqueeze(1).to(weight_dtype)

                depth = batch['depth']
                depth_stacked = depth.repeat(1,3,1,1)
                depth_resized = resize_max_res_tensor(depth_stacked, mode='depth') 
                depth_resized_normalized = depth_scale_shift_normalization(depth_resized)
                #print("depth shape:",depth.shape)
                #print("depth stack shape:",depth_stacked.shape)
                #print("depth resized shape:",depth_resized.shape)
                #print("depth resized normalized shape:",depth_resized_normalized.shape)

                normal = batch['normal'].clip(-1., 1.)
                normal_resized = resize_max_res_tensor(normal, mode='normal')
                #print("normal shape:",normal.shape)
                #print("normal resized shape:",normal_resized.shape)

                # add 
                albedo = batch['albedo'].clip(-1., 1.)
                albedo_resized = resize_max_res_tensor(albedo, mode='albedo')

                shading = batch['shading'].clip(-1., 1.)
                shading_resized = resize_max_res_tensor(shading, mode='shading')

                # encode latents
                #h_batch = vae.encoder(torch.cat((image_data_resized, depth_resized_normalized, normal_resized), dim=0).to(weight_dtype))
                h_batch = vae.encoder(torch.cat((image_data_resized, depth_resized_normalized, normal_resized,
                                                    albedo_resized, shading_resized), dim=0).to(weight_dtype))
                moments_batch = vae.quant_conv(h_batch)
                mean_batch, logvar_batch = torch.chunk(moments_batch, 2, dim=1)
                batch_latents = mean_batch * vae.config.scaling_factor
                #rgb_latents, depth_latents, normal_latents = torch.chunk(batch_latents, 3, dim=0)
                #geo_latents = torch.cat((depth_latents, normal_latents), dim=0)
                rgb_latents, depth_latents, normal_latents, albedo_latents, shading_latents = torch.chunk(batch_latents, 5, dim=0)
                geo_latents = torch.cat((depth_latents, normal_latents, albedo_latents, shading_latents), dim=0)

                # here is the setting batch size, in our settings, it can be 1.0
                bsz = rgb_latents.shape[0]
            
                # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denosing.
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=depth_latents.device).repeat(4) # 2
                timesteps = timesteps.long()

                # Sample noise that we'll add to the latents
                noise = pyramid_noise_like(geo_latents, timesteps) # create multi-res. noise
                
                # add noise to the depth lantents
                noisy_geo_latents = noise_scheduler.add_noise(geo_latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(geo_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                #batch_imgs_embed = imgs_embed.repeat((2, 1, 1))  # [B*2, 1, 768] 2
                batch_imgs_embed = imgs_embed.repeat((4, 1, 1))  # [B*2, 1, 768] 2
                
                # hybrid hierarchical switcher 
                #geo_class = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=device)
                #geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1).repeat_interleave(bsz, 0)
                geo_class = torch.tensor([[0, 0, 0,1], [0, 0, 1,0], [0,1,0,0],[1,0,0,0]], dtype=weight_dtype, device=device)
                geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1).repeat_interleave(bsz, 0)

                domain_class = batch['domain'].to(weight_dtype)
                domain_embedding = torch.cat([torch.sin(domain_class), torch.cos(domain_class)], dim=-1).repeat(4,1)# 2
                #print("geo_embedding shape:",geo_embedding.shape)
                #print("domain_embedding shape:",domain_embedding.shape)

                class_embedding = torch.cat((geo_embedding, domain_embedding), dim=-1)

                # predict the noise residual and compute the loss.
                unet_input = torch.cat((rgb_latents.repeat(4,1,1,1), noisy_geo_latents), dim=1) #2
                #print("unet input shape:",unet_input.shape)
                #print("timesteps shape:",timesteps.shape)

                topk = (
                    noise_scheduler.config.num_train_timesteps
                    // args.num_ddim_timesteps
                )
                index = torch.randint(
                    0, args.num_ddim_timesteps, (bsz,), device=geo_latents.device
                ).long()

                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(
                    timesteps < 0, torch.zeros_like(timesteps), timesteps
                )

                inference_indices = np.linspace(
                    0, len(solver.ddim_timesteps), num=args.multiphase, endpoint=False
                )
                inference_indices = np.floor(inference_indices).astype(np.int64)
                inference_indices = (
                    torch.from_numpy(inference_indices).long().to(timesteps.device)
                )
                # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions_online(
                    index, inference_indices
                )
                #[bsz,1,1,1]
                c_skip_start, c_out_start = [
                    append_dims(x,unet_input.ndim) for x in [c_skip_start, c_out_start]
                ]
                c_skip, c_out = scalings_for_boundary_conditions_target(
                    index, inference_indices
                )
                c_skip, c_out = [append_dims(x, unet_input.ndim) for x in [c_skip, c_out]]

                # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape(bsz, 1, 1, 1)
                w = w.to(device=unet_input.device, dtype=unet_input.dtype)
                #unet should be replaced
                noise_pred = unet(unet_input, 
                                timesteps, 
                                encoder_hidden_states=batch_imgs_embed,
                                class_labels=class_embedding).sample

                # epsilon_reconstruction_pred = noise_pred
                # x0_reconstruction_pred = predicted_origin(
                #     noise_pred,
                #     start_timesteps,
                #     noisy_model_input,
                #     noise_scheduler.config.prediction_type,
                #     alpha_schedule,
                #     sigma_schedule,
                # )

                pred_x_0 = predicted_origin(
                    noise_pred,
                    start_timesteps,
                    unet_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred, end_timesteps = solver.ddim_style_multiphase_pred(
                    pred_x_0, noise_pred, index, args.multiphase
                )
                model_pred = c_skip_start * unet_input + c_out_start * model_pred

                adv_timesteps = torch.empty_like(end_timesteps)

                for i in range(end_timesteps.size(0)):
                    adv_timesteps[i] = torch.randint(
                        end_timesteps[i].item(),
                        end_timesteps[i].item()
                        + noise_scheduler.config.num_train_timesteps // args.multiphase,
                        (1,),
                        dtype=end_timesteps.dtype,
                        device=end_timesteps.device,
                    )

                fake_adv = noise_scheduler.noise_travel(
                    model_pred, torch.randn_like(unet_input), end_timesteps, adv_timesteps
                )

                # 20.4.10. Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
                # noisy_latents with both the conditioning embedding c and unconditional embedding 0
                # Get teacher model prediction on noisy_latents and conditional embedding

                with torch.no_grad():
                    with torch.autocast("cuda"):
                        cond_teacher_output = teacher_unet(
                            unet_input.float(),
                            start_timesteps,
                            encoder_hidden_states=batch_imgs_embed,
                            class_labels=class_embedding,
                        ).sample
                        cond_pred_x0 = predicted_origin(
                            cond_teacher_output,
                            start_timesteps,
                            unet_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        #do not apply
                        if args.not_apply_cfg_solver:
                            uncond_teacher_output = cond_teacher_output
                            uncond_pred_x0 = cond_pred_x0
                        else:
                            # Get teacher model prediction on noisy_latents and unconditional embedding
                            uncond_teacher_output = teacher_unet(
                                unet_input.float(),
                                start_timesteps,
                                encoder_hidden_states=uncond_prompt_embeds.float(),
                            ).sample
                            uncond_pred_x0 = predicted_origin(
                                uncond_teacher_output,
                                start_timesteps,
                                unet_input,
                                noise_scheduler.config.prediction_type,
                                alpha_schedule,
                                sigma_schedule,
                            )
                        # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_teacher_output + w * (
                            cond_teacher_output - uncond_teacher_output
                        )
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                                        # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_noise_pred = unet(
                            x_prev.float(),
                            timesteps,
                            encoder_hidden_states=batch_imgs_embed,
                            class_labels=class_embedding,
                        ).sample

                    pred_x_0 = predicted_origin(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target, end_timesteps = solver.ddim_style_multiphase_pred(
                        pred_x_0, target_noise_pred, index, args.multiphase
                    )
                    target = c_skip * x_prev + c_out * target


                if global_step % 2 == 0:
                    optimizer_discriminator.zero_grad(set_to_none=True)

                    # adversarial consistency loss
                    real_adv = noise_scheduler.noise_travel(
                        target.float(), torch.randn_like(unet_input), end_timesteps, adv_timesteps
                    )
                    #discriminator
                    loss = discriminator(
                        "d_loss",
                        fake_adv.float(),
                        real_adv.float(),
                        adv_timesteps,
                        batch_imgs_embed.float(),
                        1.0,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            discriminator.parameters(), args.max_grad_norm
                        )
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad(set_to_none=True)

                else:
                    # 20.4.13. Calculate loss
                    if args.loss_type == "l2":
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    elif args.loss_type == "huber":
                        loss = torch.mean(
                            torch.sqrt(
                                (model_pred.float() - target.float()) ** 2
                                + args.huber_c**2
                            )
                            - args.huber_c
                        )

                    g_loss = args.adv_weight * discriminator(
                        "g_loss",
                        fake_adv.float(),
                        adv_timesteps,
                        batch_imgs_embed.float(),
                        1.0,
                    )
                    loss += g_loss

                    # 20.4.14. Backpropagate on the online student model (`unet`)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            unet.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            cfg=1,
                            num_inference_step=args.multiphase,
                        )
                        log_validation(
                            vae,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            cfg=7.5,
                            num_inference_step=args.multiphase,
                        )
            if (global_step - 1) % 2 == 0:
                logs = {
                    "d_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            else:
                logs = {
                    "loss_cm": loss.detach().item() - g_loss.detach().item(),
                    "g_loss": g_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
        StableDiffusionPipeline.save_lora_weights(
            os.path.join(args.output_dir, "unet_lora"), lora_state_dict
        )

    accelerator.end_training()

    #             image, text = batch

    #             image = image.to(accelerator.device, non_blocking=True)
    #             encoded_text = compute_embeddings_fn(text)

    #             pixel_values = image.to(dtype=weight_dtype)
    #             if vae.dtype != weight_dtype:
    #                 vae.to(dtype=weight_dtype)

    #             # encode pixel values with batch size of at most 32
    #             latents = []
    #             for i in range(0, pixel_values.shape[0], 32):
    #                 latents.append(
    #                     vae.encode(pixel_values[i : i + 32]).latent_dist.sample()
    #                 )
    #             #[bsz,4,64,64]
    #             latents = torch.cat(latents, dim=0)

    #             latents = latents * vae.config.scaling_factor
    #             latents = latents.to(weight_dtype)

    #             # Sample noise that we'll add to the latents
    #             noise = torch.randn_like(latents)
    #             bsz = latents.shape[0]

    #             # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
    #             topk = (
    #                 noise_scheduler.config.num_train_timesteps
    #                 // args.num_ddim_timesteps
    #             )
    #             index = torch.randint(
    #                 0, args.num_ddim_timesteps, (bsz,), device=latents.device
    #             ).long()

    #             start_timesteps = solver.ddim_timesteps[index]
    #             timesteps = start_timesteps - topk
    #             timesteps = torch.where(
    #                 timesteps < 0, torch.zeros_like(timesteps), timesteps
    #             )

    #             inference_indices = np.linspace(
    #                 0, len(solver.ddim_timesteps), num=args.multiphase, endpoint=False
    #             )
    #             inference_indices = np.floor(inference_indices).astype(np.int64)
    #             inference_indices = (
    #                 torch.from_numpy(inference_indices).long().to(timesteps.device)
    #             )
    #             # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
    #             c_skip_start, c_out_start = scalings_for_boundary_conditions_online(
    #                 index, inference_indices
    #             )
    #             #[bsz,1,1,1]
    #             c_skip_start, c_out_start = [
    #                 append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]
    #             ]
    #             c_skip, c_out = scalings_for_boundary_conditions_target(
    #                 index, inference_indices
    #             )
    #             c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

    #             #  Debug
    #             # if accelerator.is_main_process:
    #             #     print("index", index.flatten())
    #             #     print("c_skip_start", c_skip_start.flatten())
    #             #     print("c_out_start", c_out_start.flatten())
    #             #     print("c_skip", c_skip.flatten())
    #             #     print("c_out", c_out.flatten())

    #             # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
    #             # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
    #             noisy_model_input = noise_scheduler.add_noise(
    #                 latents, noise, start_timesteps
    #             )

    #             # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
    #             w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
    #             w = w.reshape(bsz, 1, 1, 1)
    #             w = w.to(device=latents.device, dtype=latents.dtype)

    #             # 20.4.8. Prepare prompt embeds and unet_added_conditions
    #             prompt_embeds = encoded_text.pop("prompt_embeds")

    #             # 20.4.9. Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
    #             # print(encoded_text) # {}
    #             #[bsz,4,64,64]
    #             noise_pred = unet(
    #                 noisy_model_input,
    #                 start_timesteps,
    #                 timestep_cond=None,
    #                 encoder_hidden_states=prompt_embeds.float(), 
    #                 added_cond_kwargs=encoded_text,
    #             ).sample

    #             epsilon_reconstruction_pred = noise_pred
    #             x0_reconstruction_pred = predicted_origin(
    #                 noise_pred,
    #                 start_timesteps,
    #                 noisy_model_input,
    #                 noise_scheduler.config.prediction_type,
    #                 alpha_schedule,
    #                 sigma_schedule,
    #             )

    #             pred_x_0 = predicted_origin(
    #                 noise_pred,
    #                 start_timesteps,
    #                 noisy_model_input,
    #                 noise_scheduler.config.prediction_type,
    #                 alpha_schedule,
    #                 sigma_schedule,
    #             )

    #             model_pred, end_timesteps = solver.ddim_style_multiphase_pred(
    #                 pred_x_0, noise_pred, index, args.multiphase
    #             )
    #             model_pred = c_skip_start * noisy_model_input + c_out_start * model_pred

    #             adv_timesteps = torch.empty_like(end_timesteps)

    #             for i in range(end_timesteps.size(0)):
    #                 adv_timesteps[i] = torch.randint(
    #                     end_timesteps[i].item(),
    #                     end_timesteps[i].item()
    #                     + noise_scheduler.config.num_train_timesteps // args.multiphase,
    #                     (1,),
    #                     dtype=end_timesteps.dtype,
    #                     device=end_timesteps.device,
    #                 )

    #             real_adv = noise_scheduler.add_noise(
    #                 latents, torch.randn_like(latents), adv_timesteps
    #             ) # not used. 
    #             fake_adv = noise_scheduler.noise_travel(
    #                 model_pred, torch.randn_like(latents), end_timesteps, adv_timesteps
    #             )

    #             # 20.4.10. Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
    #             # noisy_latents with both the conditioning embedding c and unconditional embedding 0
    #             # Get teacher model prediction on noisy_latents and conditional embedding

    #             with torch.no_grad():
    #                 with torch.autocast("cuda"):
    #                     cond_teacher_output = teacher_unet(
    #                         noisy_model_input.float(),
    #                         start_timesteps,
    #                         encoder_hidden_states=prompt_embeds.float(),
    #                     ).sample
    #                     cond_pred_x0 = predicted_origin(
    #                         cond_teacher_output,
    #                         start_timesteps,
    #                         noisy_model_input,
    #                         noise_scheduler.config.prediction_type,
    #                         alpha_schedule,
    #                         sigma_schedule,
    #                     )
    #                     if args.not_apply_cfg_solver:
    #                         uncond_teacher_output = cond_teacher_output
    #                         uncond_pred_x0 = cond_pred_x0
    #                     else:
    #                         # Get teacher model prediction on noisy_latents and unconditional embedding
    #                         uncond_teacher_output = teacher_unet(
    #                             noisy_model_input.float(),
    #                             start_timesteps,
    #                             encoder_hidden_states=uncond_prompt_embeds.float(),
    #                         ).sample
    #                         uncond_pred_x0 = predicted_origin(
    #                             uncond_teacher_output,
    #                             start_timesteps,
    #                             noisy_model_input,
    #                             noise_scheduler.config.prediction_type,
    #                             alpha_schedule,
    #                             sigma_schedule,
    #                         )
    #                     # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
    #                     pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
    #                     pred_noise = cond_teacher_output + w * (
    #                         cond_teacher_output - uncond_teacher_output
    #                     )
    #                     x_prev = solver.ddim_step(pred_x0, pred_noise, index)

    #             # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
    #             with torch.no_grad():
    #                 with torch.autocast("cuda", dtype=weight_dtype):
    #                     target_noise_pred = unet(
    #                         x_prev.float(),
    #                         timesteps,
    #                         timestep_cond=None,
    #                         encoder_hidden_states=prompt_embeds.float(),
    #                     ).sample

    #                 pred_x_0 = predicted_origin(
    #                     target_noise_pred,
    #                     timesteps,
    #                     x_prev,
    #                     noise_scheduler.config.prediction_type,
    #                     alpha_schedule,
    #                     sigma_schedule,
    #                 )
    #                 target, end_timesteps = solver.ddim_style_multiphase_pred(
    #                     pred_x_0, target_noise_pred, index, args.multiphase
    #                 )
    #                 target = c_skip * x_prev + c_out * target


    #             if global_step % 2 == 0:
    #                 optimizer_discriminator.zero_grad(set_to_none=True)

    #                 # adversarial consistency loss
    #                 real_adv = noise_scheduler.noise_travel(
    #                     target.float(), torch.randn_like(latents), end_timesteps, adv_timesteps
    #                 )

    #                 loss = discriminator(
    #                     "d_loss",
    #                     fake_adv.float(),
    #                     real_adv.float(),
    #                     adv_timesteps,
    #                     prompt_embeds.float(),
    #                     1.0,
    #                 )
    #                 accelerator.backward(loss)
    #                 if accelerator.sync_gradients:
    #                     accelerator.clip_grad_norm_(
    #                         discriminator.parameters(), args.max_grad_norm
    #                     )
    #                 optimizer_discriminator.step()
    #                 optimizer_discriminator.zero_grad(set_to_none=True)

    #             else:
    #                 # 20.4.13. Calculate loss
    #                 if args.loss_type == "l2":
    #                     loss = F.mse_loss(
    #                         model_pred.float(), target.float(), reduction="mean"
    #                     )
    #                 elif args.loss_type == "huber":
    #                     loss = torch.mean(
    #                         torch.sqrt(
    #                             (model_pred.float() - target.float()) ** 2
    #                             + args.huber_c**2
    #                         )
    #                         - args.huber_c
    #                     )

    #                 g_loss = args.adv_weight * discriminator(
    #                     "g_loss",
    #                     fake_adv.float(),
    #                     adv_timesteps,
    #                     prompt_embeds.float(),
    #                     1.0,
    #                 )
    #                 loss += g_loss

    #                 # 20.4.14. Backpropagate on the online student model (`unet`)
    #                 accelerator.backward(loss)
    #                 if accelerator.sync_gradients:
    #                     accelerator.clip_grad_norm_(
    #                         unet.parameters(), args.max_grad_norm
    #                     )
    #                 optimizer.step()
    #                 lr_scheduler.step()
    #                 optimizer.zero_grad(set_to_none=True)

    #         # Checks if the accelerator has performed an optimization step behind the scenes
    #         if accelerator.sync_gradients:
    #             progress_bar.update(1)
    #             global_step += 1

    #             if accelerator.is_main_process:
    #                 if global_step % args.checkpointing_steps == 0:
    #                     # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    #                     if args.checkpoints_total_limit is not None:
    #                         checkpoints = os.listdir(args.output_dir)
    #                         checkpoints = [
    #                             d for d in checkpoints if d.startswith("checkpoint")
    #                         ]
    #                         checkpoints = sorted(
    #                             checkpoints, key=lambda x: int(x.split("-")[1])
    #                         )

    #                         # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    #                         if len(checkpoints) >= args.checkpoints_total_limit:
    #                             num_to_remove = (
    #                                 len(checkpoints) - args.checkpoints_total_limit + 1
    #                             )
    #                             removing_checkpoints = checkpoints[0:num_to_remove]

    #                             logger.info(
    #                                 f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
    #                             )
    #                             logger.info(
    #                                 f"removing checkpoints: {', '.join(removing_checkpoints)}"
    #                             )

    #                             for removing_checkpoint in removing_checkpoints:
    #                                 removing_checkpoint = os.path.join(
    #                                     args.output_dir, removing_checkpoint
    #                                 )
    #                                 shutil.rmtree(removing_checkpoint)

    #                     save_path = os.path.join(
    #                         args.output_dir, f"checkpoint-{global_step}"
    #                     )
    #                     accelerator.save_state(save_path)
    #                     logger.info(f"Saved state to {save_path}")

    #                 if global_step % args.validation_steps == 0:
    #                     log_validation(
    #                         vae,
    #                         unet,
    #                         args,
    #                         accelerator,
    #                         weight_dtype,
    #                         global_step,
    #                         cfg=1,
    #                         num_inference_step=args.multiphase,
    #                     )
    #                     log_validation(
    #                         vae,
    #                         unet,
    #                         args,
    #                         accelerator,
    #                         weight_dtype,
    #                         global_step,
    #                         cfg=7.5,
    #                         num_inference_step=args.multiphase,
    #                     )
    #         if (global_step - 1) % 2 == 0:
    #             logs = {
    #                 "d_loss": loss.detach().item(),
    #                 "lr": lr_scheduler.get_last_lr()[0],
    #             }
    #         else:
    #             logs = {
    #                 "loss_cm": loss.detach().item() - g_loss.detach().item(),
    #                 "g_loss": g_loss.detach().item(),
    #                 "lr": lr_scheduler.get_last_lr()[0],
    #             }
    #         progress_bar.set_postfix(**logs)
    #         accelerator.log(logs, step=global_step)

    #         if global_step >= args.max_train_steps:
    #             break

    # # Create the pipeline using using the trained modules and save it.
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = accelerator.unwrap_model(unet)
    #     unet.save_pretrained(args.output_dir)
    #     lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
    #     StableDiffusionPipeline.save_lora_weights(
    #         os.path.join(args.output_dir, "unet_lora"), lora_state_dict
    #     )

    # accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

