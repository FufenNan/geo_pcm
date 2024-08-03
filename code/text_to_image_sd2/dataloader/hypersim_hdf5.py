import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import glob
from tqdm import tqdm
import imgaug.augmenters as iaa

import h5py
import numpy as np

def read_hdf5_to_numpy(file_path, dataset_name = 'dataset'):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

class MixDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(MixDataset, self).__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.img_size = (576, 768)
        self.samples = []

        # Load Hypersim dataset
        self.load_hypersim()

    def load_hypersim(self):
        scene_dirs = glob.glob(os.path.join(self.data_dir, 'ai_*'))
        for scene_dir in tqdm(scene_dirs, desc='Loading HyperSim'):
            final_preview_dir = os.path.join(scene_dir, 'images', 'scene_cam_*_final_hdf5')
            geometry_preview_dir = os.path.join(scene_dir, 'images', 'scene_cam_*_geometry_hdf5')
            for image_dir in glob.glob(final_preview_dir):
                color_files = glob.glob(os.path.join(image_dir, 'frame.*.color.hdf5'))
                for color_file in color_files:
                    base_name = os.path.basename(color_file).replace('.color.hdf5', '')

                    depth_file = os.path.join(image_dir.replace('final_hdf5', 'geometry_hdf5'), f'{base_name}.depth_meters.hdf5')
                    normal_file = os.path.join(image_dir.replace('final_hdf5', 'geometry_hdf5'), f'{base_name}.normal_cam.hdf5')

                    # Albedo and Shading files in the final_hdf5 folder
                    albedo_file = os.path.join(image_dir, f'{base_name}.diffuse_reflectance.hdf5')
                    shading_file = os.path.join(image_dir, f'{base_name}.diffuse_illumination.hdf5')

                    #if not os.path.exists(depth_file) or not os.path.exists(normal_file):
                        #continue

                    sample = {
                        'dataset': 'hypersim',
                        'rgb': color_file,
                        'depth': depth_file,
                        'normal': normal_file,
                        'albedo': albedo_file,
                        'shading': shading_file,
                        'is_complete': True,
                        'RandomHorizontalFlip': 0.4,
                        'distortion_prob': 0.05,
                        'to_gray_prob': 0.1
                    }
                    self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        H, W = self.img_size

        # HyperSim
        if sample_path['dataset'] == 'hypersim':
            sample['domain'] = torch.Tensor([1., 0., 0.])  # indoor

            '''
            sample['rgb'] = cv2.imread(sample_path['rgb'])[:, :, ::-1]  # [H, W, 3]
            sample['depth'] = cv2.imread(sample_path['depth'])[:, :, 0].astype(np.float32)  # [H, W, 3]
            sample['normal'] = cv2.imread(sample_path['normal'])[:, :, ::-1].astype(np.float32)  # [H, W, 3] # [H, W, 3]
            sample['albedo'] = cv2.imread(sample_path['albedo'])[:, :, 0].astype(np.float32)  # [H, W, 3]
            sample['shading'] = cv2.imread(sample_path['shading'])[:, :, ::-1].astype(np.float32)  # [H, W, 3] # [H, W, 3]
            '''
            sample['rgb'] = read_hdf5_to_numpy(sample_path['rgb'])#.astype(np.float64)
            sample['depth'] = read_hdf5_to_numpy(sample_path['depth'])#.astype(np.float64)
            sample['normal'] = read_hdf5_to_numpy(sample_path['normal'])#.astype(np.float64)
            sample['albedo'] = read_hdf5_to_numpy(sample_path['albedo'])#.astype(np.float64)
            sample['shading'] = read_hdf5_to_numpy(sample_path['shading'])#.astype(np.float64)
            sample['rgb'] = (sample['rgb'] * 255).astype(np.uint8)

            H_ori, W_ori = sample['rgb'].shape[:2]
            #print(f'H_ori:{H_ori}, W_ori:{W_ori}')

            # ----------------- Data Augmentation -----------------

            # 1. Random Crop

            if H_ori >= H and W_ori >= W:
                H_start, W_start = np.random.randint(0, H_ori-H+1), np.random.randint(0, W_ori-W+1)
                sample['rgb'] = sample['rgb'][H_start:H_start + H, W_start:W_start + W]
                sample['depth'] = sample['depth'][H_start:H_start + H, W_start:W_start + W]
                sample['normal'] = sample['normal'][H_start:H_start + H, W_start:W_start + W]
                sample['albedo'] = sample['albedo'][H_start:H_start + H, W_start:W_start + W]
                sample['shading'] = sample['shading'][H_start:H_start + H, W_start:W_start + W]

            # 2. Random Horizontal Flip
            if np.random.random() < sample_path['RandomHorizontalFlip']:
                sample['rgb'] = np.copy(np.fliplr(sample['rgb']))
                sample['depth'] = np.copy(np.fliplr(sample['depth']))
                sample['normal'] = np.copy(np.fliplr(sample['normal']))
                sample['albedo'] = np.copy(np.fliplr(sample['albedo']))
                sample['shading'] = np.copy(np.fliplr(sample['shading']))
                sample['normal'][:, :, 0] *= -1.
                #print(f'albedo shape:{sample["albedo"].shape}')
                #print(f'shading shape:{sample["shading"].shape}')

            # 3. Photometric Distortion
            to_gray_prob = sample_path['to_gray_prob']
            distortion_prob = sample_path['distortion_prob']
            brightness_beta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.5, 1.5)
            saturate_alpha = np.random.uniform(0.5, 1.5)
            rand_hue = np.random.randint(-18, 18)

            brightness_do = np.random.random() < distortion_prob
            contrast_do = np.random.random() < distortion_prob
            saturate_do = np.random.random() < distortion_prob
            rand_hue_do = np.random.random() < distortion_prob

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = 0 if np.random.random() > 0.5 else 1
            if np.random.random() < to_gray_prob:
                sample['rgb'] = iaa.Grayscale(alpha=(0.8, 1.0))(image=sample['rgb'])
            else:
                # random brightness
                if brightness_do:
                    alpha, beta = 1.0, brightness_beta
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float64) * alpha + beta), 0, 255).astype(np.uint8)

                if mode == 0:
                    if contrast_do:
                        alpha, beta = contrast_alpha, 0.0
                        sample['rgb'] = np.clip((sample['rgb'].astype(np.float64) * alpha + beta), 0, 255).astype(np.uint8)

                '''
                # random saturation
                if saturate_do:
                    img = cv2.cvtColor(sample['rgb'][:, :, ::-1], cv2.COLOR_BGR2HSV)
                    alpha, beta = saturate_alpha, 0.0
                    img[:, :, 1] = np.clip((img[:, :, 1].astype(np.float64) * alpha + beta), 0, 255).astype(np.uint8)
                    sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:, :, ::-1]

                # random hue
                if rand_hue_do:
                    img = cv2.cvtColor(sample['rgb'][:, :, ::-1], cv2.COLOR_BGR2HSV)
                    img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
                    sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:, :, ::-1]
                '''
                if saturate_do:
                    img = sample['rgb'].astype(np.float32)
                    img = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2HSV)
                    alpha, beta = saturate_alpha, 0.0
                    img[:, :, 1] = np.clip((img[:, :, 1] * alpha + beta), 0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:, :, ::-1]
                    sample['rgb'] = img.astype(np.float64)

                if rand_hue_do:
                    img = sample['rgb'].astype(np.float32)
                    img = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2HSV)
                    img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
                    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:, :, ::-1]
                    sample['rgb'] = img.astype(np.float64)

                # random contrast
                if mode == 1:
                    if contrast_do:
                        alpha, beta = contrast_alpha, 0.0
                        sample['rgb'] = np.clip((sample['rgb'].astype(np.float64) * alpha + beta), 0, 255).astype(np.uint8)

            # 4. To Tensor
            sample['rgb'] = ((torch.from_numpy(np.transpose(sample['rgb'].copy(), (2, 0, 1))) / 255.) * 2.0 - 1.0)#.to(torch.unit8)  # [3, H, W]
            sample['depth'] = torch.from_numpy(sample['depth'][None].copy())#.to(torch.float64)  # [1, H, W]
            sample['normal'] = torch.from_numpy(np.transpose(sample['normal'].copy(), (2, 0, 1)))#.to(torch.float64)  # [3, H, W]
            sample['albedo'] = torch.from_numpy(np.transpose(sample['albedo'].copy(), (2, 0, 1)))#.to(torch.float64)  # [3, H, W]
            sample['shading'] = torch.from_numpy(np.transpose(sample['shading'].copy(), (2, 0, 1)))#.to(torch.float64)  # [3, H, W]

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size
