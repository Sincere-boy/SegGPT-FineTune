# SegGPT Fine-Tune
**Unofficial** training code for [SegGPT](https://github.com/baaivision/Painter).

![SegGPT](https://i.imgur.com/ZqxJzmI.png)
From left to right: input image, masked label, ground truth, raw model prediction, discretized model prediction.

## Disclaimer
- This implementation is based on **my understanding of the paper** and by **reverse-engineering how the model works** from the official implementation.
- I have tested this code to fine-tune on [OEM dataset](https://open-earth-map.org/) and got promising results. However, there might be some bugs or mistakes in the code. Feel free to raise an issue.
- Fine-tuning from the provided [checkpoint](https://huggingface.co/BAAI/SegGPT/blob/main/seggpt_vit_large.pth) requires a lot of GPU memory (at least 24GB) as this trains the whole ViT-16 backbone. Consider using smaller batch size or smaller model overall. I might implement training using LoRA to support smaller VRAM in the future if this repo gains enough tractions.

## Setup
This code is developed with Python 3.9.

### PIP
Install the required packages by running:
```bash
pip install -r requirements.txt
```

### Conda (Only for Linux)
Create a new conda environment and install the required packages by running:
```bash
conda env create -f env.yml
```

## Dataset
Setup your dataset directory as follows:
```
<root_dataset_path>
├── images
│   ├── image1.tif
│   ├── image2.tif
│   ...
└── labels
    ├── image1.tif
    ├── image2.tif
    ...
```
**Note**:
- Image and labels must have the same name and extension (or you can modify `data.py` to support your needs).
- The extension **does not** have to be `.tif` as long as it can be loaded using `PIL` library. 
- The label is a **single-channel** image where each pixel value represents the **class** of that pixel.

## Training
Create a `.json` config file. You can use the provided `configs/base.json` as a template. Then, run:
```bash
python train.py --config <path_to_json_config>
```
The training uses DDP strategy and utilizes all available GPUs by default. You can specify the GPU to use by setting `CUDA_VISIBLE_DEVICES` in the environment variable.

### Training a pruned model
If you have pruned SegGPT and saved it with `show_cosine.py`, set the `pruned_dir` field in the config to the directory containing the pruned weights (`pytorch_model.bin` and `keep_indices.json`). The training script will load this structure instead of the full model.

You can also launch tensorboard to monitor the training progress:
```bash
tensorboard --logdir logs
```

## Learnable Tensor
In the paper, the author mentioned using learnable tensor for in-context tuning. You can find my implementation for this in `model.py`.