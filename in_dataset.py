import random
import os
import csv
from PIL import Image
import torch.utils.data as data_utils
from collections import defaultdict
import torchvision.transforms.functional as tf
import torch


class INDataset(data_utils.Dataset):

    def __init__(self, img_root, dataset, args, transform=None, task='S', fold=0):
        super().__init__()

        self.img_root = img_root
        self.dataset = dataset
        self.args = args
        self.transform = transform
        self.task = task
        self.fold = fold

        self.data = []          # img_id
        self.label = []         # mapped label
        self.img_map = {}       # img_id -> {"C": fname, "G": fname}

        # ===== 扫描 train 目录 =====
        train_dir = os.path.join(img_root, "train")
        for fname in os.listdir(train_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                parts = fname.split("-")
                img_id = parts[0]
                mode = parts[2].split(".")[0]  # C or G
                if img_id not in self.img_map:
                    self.img_map[img_id] = {}
                self.img_map[img_id][mode] = fname

        # ===== 读取 CSV =====
        if dataset == 'train':
            folds = [i for i in range(5) if i != fold]
        else:
            folds = [fold]

        for f in folds:
            csv_path = os.path.join(img_root, f"fold{f}.csv")
            with open(csv_path) as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    img_id = row[0]
                    raw_label = int(row[1])

                    if img_id not in self.img_map:
                        continue
                    if "C" not in self.img_map[img_id] or "G" not in self.img_map[img_id]:
                        continue

                    self.data.append(img_id)
                    self.label.append(raw_label)

        self._map_labels()

        # ===== class index（few-shot用）=====
        self.class_to_indices = defaultdict(list)
        for idx, lab in enumerate(self.label):
            self.class_to_indices[lab].append(idx)

        print("Label distribution:", {k: len(v) for k, v in self.class_to_indices.items()})
        print("Total samples:", len(self.data))

    # ===============================
    # label remap
    # ===============================
    def _map_labels(self):
        new_data, new_label = [], []
        for img_id, raw_label in zip(self.data, self.label):

            if self.task == 'S':
                new_data.append(img_id)
                new_label.append(raw_label)

            elif self.task == 'T1':
                new_data.append(img_id)
                new_label.append(0 if raw_label == 0 else 1)

            elif self.task == 'T2':
                if raw_label in [1, 2]:
                    new_data.append(img_id)
                    new_label.append(raw_label - 1)

        self.data = new_data
        self.label = new_label

    # ===============================
    # 同步增强（C/G一起变换）
    # ===============================
    def get_aug_img(self, img1, img2):

        img1 = tf.resize(img1, [224, 224])
        img2 = tf.resize(img2, [224, 224])

        if random.random() > 0.5:
            img1 = tf.hflip(img1)
            img2 = tf.hflip(img2)

        if random.random() > 0.5:
            img1 = tf.vflip(img1)
            img2 = tf.vflip(img2)

        if random.random() > 0.3:
            degree = random.uniform(-10, 10)
            img1 = tf.rotate(img1, degree)
            img2 = tf.rotate(img2, degree)

        # ---------- 只在是 PIL 时才转 tensor ----------
        if not isinstance(img1, torch.Tensor):
            img1 = tf.to_tensor(img1)
        if not isinstance(img2, torch.Tensor):
            img2 = tf.to_tensor(img2)

        img1 = tf.normalize(img1, mean=[0.46]*3, std=[0.1582]*3)
        img2 = tf.normalize(img2, mean=[0.46]*3, std=[0.1582]*3)

        return img1, img2

    # ===============================
    # 基础接口
    # ===============================
    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.label

    def __getitem__(self, idx):
        img_id = self.data[idx]
        label = self.label[idx]

        fname_C = self.img_map[img_id]["C"]
        fname_G = self.img_map[img_id]["G"]

        path_C = os.path.join(self.img_root, "train", fname_C)
        path_G = os.path.join(self.img_root, "train", fname_G)

        img_C = Image.open(path_C).convert("RGB")
        img_G = Image.open(path_G).convert("RGB")

        img_C, img_G = self.get_aug_img(img_C, img_G)
        return img_C, label, img_G


import torch
from torch.utils.data import Dataset
import random

class INTestDataset(Dataset):
    """Few-shot validation/test dataset with separate support and query sets."""

    def __init__(self, support_dataset, query_dataset, args):
        super().__init__()
        self.support_dataset = support_dataset  # train dataset, 用于 support
        self.query_dataset = query_dataset      # val dataset, 用于 query
        self.args = args

        # 支持集合按类别索引
        self.class_to_indices = support_dataset.class_to_indices
        # query 总长度就是 val dataset 样本数
        self.num_episodes = len(query_dataset)

    def __len__(self):
        return self.num_episodes

    def get_support_set(self):
        """按类别从 support_dataset 中抽取 shot 张每类"""
        support_C, support_G, support_label = [], [], []

        selected_classes = list(self.class_to_indices.keys())

        for new_label, cls in enumerate(selected_classes):
            indices = self.class_to_indices[cls]
            chosen = random.sample(indices, self.args.shot)  # 每类抽 shot 张
            for idx in chosen:
                img_C, label, img_G = self.support_dataset[idx]
                support_C.append(img_C)
                support_G.append(img_G)
                support_label.append(new_label)

        # 转 tensor
        support_C = torch.stack(support_C)
        support_G = torch.stack(support_G)
        support_label = torch.tensor(support_label)

        return support_C, support_G, support_label

    def __getitem__(self, idx):
        """返回一个 episode: 当前 idx 的 query + 固定 support"""
        # 当前 query 样本
        query_C, query_label, query_G = self.query_dataset[idx]
        # 把 support 构建好
        support_C, support_G, support_label = self.get_support_set()

        # query 转 tensor
        query_C = query_C.unsqueeze(0) if query_C.dim() == 3 else query_C
        query_G = query_G.unsqueeze(0) if query_G.dim() == 3 else query_G
        query_label = torch.tensor([query_label])

        return query_C, query_label, query_G, support_C, support_label, support_G