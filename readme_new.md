## To test the model you should:
1. follow the instructions in README.md
2. change the path to datasets
   in /code/text_to_image_sd15/train_pcm_lora_sd15_adv.py
   ```python
   train_dataset = CustomImageDataset(
        #change to my dataset
        "/home/haoyum3/PCM/cc3m/data", args.resolution
    )
    ```
   if you want to test geo_pcm, set the path to hypersim in /code/text_to_image_sd2/train_adv.sh
   ```python
    --adv_weight=0.1 \
    --adv_lr=1e-5 \
    --dataset_path=
4. run /code/text_to_image_sd2/train_adv.sh or /code/text_to_image_sd15/train_adv.sh

train_pcm_lora_sd2_adv.py is runable  (the code to distill sdv2)

but I still try to debug text_to_image_sd2/train_pcm_lora_adv_decom.py (the code to distill geo_wizard)
