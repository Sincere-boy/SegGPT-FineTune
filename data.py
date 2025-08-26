# data.py
import os
import os.path as osp
from typing import Iterable, Tuple, List

import numpy as np
from PIL import Image
from itertools import combinations, permutations
from tqdm import tqdm

import torch
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from utils import get_logger


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        n_classes: int,
        mean: Iterable[float] = (0.485, 0.456, 0.406),
        std: Iterable[float] = (0.229, 0.224, 0.225),
        resize: Tuple[int, int] = (448, 448),      # (H, W)
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        is_train: bool = True,
    ):
        """
        目录结构要求：
        root/
          images/  <RGB图像，名称与labels对应>
          labels/  <灰度/索引标签，0..K-1 或二值 0/1>
        """
        super().__init__()
        assert osp.exists(osp.join(root, "images")), f"Path {root}/images does not exist"
        assert osp.exists(osp.join(root, "labels")), f"Path {root}/labels does not exist"

        self.root = root
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize = resize
        self.is_train = is_train
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.logger = get_logger(__class__.__name__, 0)  # TODO: 每个进程都会log

        self.paths = []
        for path in os.listdir(osp.join(self.root, "images")):
            img_path = osp.join(self.root, "images", path)
            label_path = osp.join(self.root, "labels", path)
            if not osp.exists(label_path):
                self.logger.warn(f"Skipping label path {label_path} as it does not exist")
                continue
            self.paths.append((img_path, label_path))

        self._preload_dataset()
        self._generate_pairs()
        self._init_augmentation()
        self._filter_pairs()

    # -------------------- 预加载与配对 --------------------
    def _preload_dataset(self):
        self.images = []
        self.labels = []
        self.unique_classes = []
        for img_path, label_path in tqdm(self.paths, desc="Caching images and labels"):
            img = self._load_img(img_path)
            label = self._load_lbl(label_path)
            self.images.append(img)
            self.labels.append(label)
            self.unique_classes.append(set(np.unique(label)))

    def _generate_pairs(self):
        indices = np.arange(len(self.paths))
        if self.is_train:
            self.pairs = list(combinations(indices, 2))
        else:
            self.pairs = list(permutations(indices, 2))

    def _filter_pairs(self):
        self.same_class_pairs = []
        self.diff_class_pairs = []
        for pair in tqdm(self.pairs, desc="Filtering pairs"):
            len_intersect = self.unique_classes[pair[0]].intersection(self.unique_classes[pair[1]])
            len_union = self.unique_classes[pair[0]].union(self.unique_classes[pair[1]])
            if len_intersect == len_union:
                self.same_class_pairs.append(pair)
            else:
                self.diff_class_pairs.append(pair)
        np.random.shuffle(self.same_class_pairs)
        np.random.shuffle(self.diff_class_pairs)

        # 在 BaseDataset 类里加上这个方法
    def _augment_pair(self,
                    imgs: List[np.ndarray],
                    ori_labels: List[np.ndarray],
                    color_palette: np.ndarray):
        det_geo = self.augment_geo.to_deterministic()

        imgs_g, ori_labels_g = [], []
        for img, ori in zip(imgs, ori_labels):
            seg = SegmentationMapsOnImage(ori.astype(np.int32), shape=ori.shape)
            img_g, seg_g = det_geo(image=img, segmentation_maps=seg)
            imgs_g.append(img_g)
            ori_labels_g.append(seg_g.get_arr().astype(np.uint8))

        # 根据增强后的 ori_label 重着色
        labels_rgb_g = [self._lbl_random_color(ol, color_palette) for ol in ori_labels_g]

        # 像素增强（只作用图像）
        det_img = self.augment_img.to_deterministic()
        imgs_i = det_img.augment_images(imgs_g)

        return imgs_i, labels_rgb_g, ori_labels_g
    # -------------------- I/O 与基本处理 --------------------
    def _load_img(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if self.resize is not None:
            img = img.resize(self.resize)  # PIL: size=(W,H)? 这里传(Tuple[int,int])，PIL按(W,H)，但你的self.resize是(H,W)。
        # 上一行如果你严格想按(H,W)→(W,H)，请改： img = img.resize((self.resize[1], self.resize[0]))
        # 为与原逻辑保持一致，这里沿用你原代码：直接传 self.resize
        img = np.array(img).astype(np.uint8)
        return img

    def _load_lbl(self, path: str) -> np.ndarray:
        label = Image.open(path).convert("L")
        if self.resize is not None:
            label = label.resize(self.resize, Image.NEAREST)
        label = np.array(label).astype(np.uint8)
        return label

    def _generate_color_palette(self) -> np.ndarray:
        # (n_classes, 3) in [0,255]
        return np.random.randint(0, 256, (self.n_classes, 3), dtype=np.uint8)


    def _init_augmentation(self):
        H, W = self.resize if self.resize is not None else (448, 448)

        if self.is_train:
            self.augment_geo = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                iaa.Affine(
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    order=1, cval=0, mode="constant",
                ),
                iaa.PerspectiveTransform(scale=(0.0, 0.06), keep_size=True),
                iaa.CropToFixedSize(W, H)
            ])

            # ⚠️ 替换这里：不再用 AddToHueAndSaturation，改为 WithColorspace + WithChannels
            self.augment_img = iaa.Sequential([
                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                iaa.LinearContrast((0.9, 1.1)),

                iaa.WithColorspace(
                    to_colorspace="HSV", from_colorspace="RGB",
                    children=iaa.Sequential([
                        iaa.WithChannels(0, iaa.Add((-8, 8))),       # Hue 微调
                        iaa.WithChannels(1, iaa.Multiply((0.9, 1.1))) # Saturation 微调
                    ])
                ),

                iaa.GaussianBlur((0.0, 0.8)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.08), per_channel=0.2),
            ])
        else:
            self.augment_geo = iaa.Sequential([iaa.CropToFixedSize(W, H)])
            self.augment_img = iaa.Sequential([])

    # -------------------- 工具 --------------------
    def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray) -> np.ndarray:
        """将索引/二值label映射到RGB伪彩色图。"""
        h, w = label.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        # 只映射 [0, n_classes-1]
        for i in range(self.n_classes):
            result[label == i] = color_palette[i]
        return result

    def _to_img_tensor(self, arr: np.ndarray) -> torch.FloatTensor:
        """ 图片/彩色label：/255 -> 标准化 -> CHW """
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        res = torch.from_numpy(arr).float()
        res = torch.einsum("hwc->chw", res)
        return res

    def _generate_mask(self, img_shape: Tuple[int, int], is_half: bool = False) -> torch.Tensor:
        """生成patch级mask。1=masked, 0=visible"""
        total_patch = (img_shape[0] // self.patch_size[0]) * (img_shape[1] // self.patch_size[1])
        if is_half:
            mask = torch.zeros(total_patch, dtype=torch.float32)
            mask[total_patch // 2:] = 1
        else:
            total_ones = int(total_patch * self.mask_ratio)
            shuffle_idx = torch.randperm(total_patch)
            mask = torch.FloatTensor([0] * (total_patch - total_ones) + [1] * total_ones)[shuffle_idx]
        return mask

    # -------------------- 关键：与原返回完全兼容 --------------------
    def __getitem__(self, idx):
        if self.is_train:
            if idx < len(self.same_class_pairs):
                pair_idx1, pair_idx2 = self.same_class_pairs[idx]
            else:
                pair_idx1, pair_idx2 = self.diff_class_pairs[idx - len(self.same_class_pairs)]
            # 随机swap
            if np.random.rand() > 0.5:
                pair_idx1, pair_idx2 = pair_idx2, pair_idx1
        else:
            pair_idx1, pair_idx2 = self.same_class_pairs[idx]

        img1, ori_label1 = self.images[pair_idx1], self.labels[pair_idx1]
        img2, ori_label2 = self.images[pair_idx2], self.labels[pair_idx2]

        # 随机调色板（每样本对）
        color_palette = self._generate_color_palette()

        # 增强（几何同步 + 像素只对图像）→ 再着色 label
        img_list, label_list, ori_label_list = self._augment_pair(
            [img1, img2], [ori_label1, ori_label2], color_palette
        )

        # 上下拼接（保持原逻辑）
        img = np.concatenate(img_list, axis=0)           # (2H, W, 3)
        label = np.concatenate(label_list, axis=0)       # (2H, W, 3)
        ori_label = np.concatenate(ori_label_list, axis=0)  # (2H, W)

        # 转tensor & 标准化（保持原逻辑）
        img = self._to_img_tensor(img)       # [3, 2H, W]
        label = self._to_img_tensor(label)   # [3, 2H, W]

        # 你的原实现将灰度阈值>127映射为1（假定二类）。如为多类，请在此改为 int64 直接返回索引。
        ori_label = (ori_label > 127).astype(np.uint8)
        ori_label = torch.LongTensor(ori_label)          # [2H, W]

        # 生成mask（验证集=half；训练集：same_class_pairs部分=half）
        is_half = (not self.is_train) or (idx < len(self.same_class_pairs))
        mask = self._generate_mask((img.shape[1], img.shape[2]), is_half)

        valid = torch.ones_like(label)
        seg_type = torch.zeros([1])
        color_palette = torch.FloatTensor(color_palette)
        return img, label, mask, valid, seg_type, ori_label, color_palette

    def __len__(self):
        if self.is_train:
            return len(self.same_class_pairs) + len(self.diff_class_pairs)
        return min(len(self.same_class_pairs), 1600)