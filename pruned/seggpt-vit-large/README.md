---
license: apache-2.0
tags:
- vision
inference: false
---

# SegGPT model

The SegGPT model was proposed in [SegGPT: Segmenting Everything In Context](https://arxiv.org/abs/2304.03284) by Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang.

## Model description

SegGPT employs a decoder-only (GPT-like) Transformer that can generate a segmentation mask given an input image, a prompt image and its corresponding prompt mask.
The model achieves remarkable one-shot results with 56.1 mIoU on COCO-20 and 85.6 mIoU on FSS-1000.

## Intended uses & limitations

You can use the raw model for one-shot image segmentation.

### How to use

Here's how to use the model for one-shot semantic segmentation:

```python
import torch
from datasets import load_dataset
from transformers import SegGptImageProcessor, SegGptForImageSegmentation

model_id = "BAAI/seggpt-vit-large"
image_processor = SegGptImageProcessor.from_pretrained(checkpoint)
model = SegGptForImageSegmentation.from_pretrained(checkpoint)

dataset_id = "EduardoPacheco/FoodSeg103"
ds = load_dataset(dataset_id, split="train")
# Number of labels in FoodSeg103 (not including background)
num_labels = 103

image_input = ds[4]["image"]
ground_truth = ds[4]["label"]
image_prompt = ds[29]["image"]
mask_prompt = ds[29]["label"]

inputs = image_processor(
    images=image_input, 
    prompt_images=image_prompt,
    prompt_masks=mask_prompt, 
    num_labels=num_labels,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = [image_input.size[::-1]]
mask = image_processor.post_process_semantic_segmentation(outputs, target_sizes, num_labels=num_labels)[0]
```

### BibTeX entry and citation info

```bibtex
@misc{wang2023seggpt,
      title={SegGPT: Segmenting Everything In Context}, 
      author={Xinlong Wang and Xiaosong Zhang and Yue Cao and Wen Wang and Chunhua Shen and Tiejun Huang},
      year={2023},
      eprint={2304.03284},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```