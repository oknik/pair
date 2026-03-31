import os
import csv
import random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data_utils
import torchvision.transforms.functional as tf


class TeacherDataset(data_utils.Dataset):

    def __init__(self, img_root, dataset, task, args=None, fold=0, is_train=True, few_shot=False, support_dataset=None, fixed_support=False):

        super().__init__()

        # 必须参数
        self.img_root = img_root
        self.dataset = dataset
        self.args = args
        self.task = task
        self.fold = fold
        self.is_train = is_train
        self.few_shot = few_shot
        self.support_dataset = support_dataset
        self.fixed_support = fixed_support

        self.data = []
        self.label = []
        self.img_map = {}

        img_dir = os.path.join(img_root, dataset)

        # ========= 扫描图像 =========
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                parts = fname.split("-")

                img_id = str(int(parts[0]))
                mode = parts[2].split(".")[0]

                if img_id not in self.img_map:
                    self.img_map[img_id] = {}

                self.img_map[img_id][mode] = fname

        # ========= 读取CSV =========
        csv_path = os.path.join(img_root, f"{dataset}.csv")

        with open(csv_path) as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                img_id = str(int(row[0]))
                raw_label = int(row[1])

                if img_id not in self.img_map:
                    continue
                if "C" not in self.img_map[img_id] or "G" not in self.img_map[img_id]:
                    continue

                self.data.append(img_id)
                self.label.append(raw_label)

        # label映射
        self._map_labels()

        # few-shot class index
        self.class_to_indices = defaultdict(list)
        for idx, lab in enumerate(self.label):
            self.class_to_indices[lab].append(idx)

        print("Label distribution:",
              {k: len(v) for k, v in self.class_to_indices.items()})
        print("Total samples:", len(self.data))

        if self.few_shot and self.fixed_support:
            self.cached_support = self._build_support_set()

    # =================================
    # label remap
    # =================================
    def _map_labels(self):
        new_data = []
        new_label = []

        for img_id, raw_label in zip(self.data, self.label):
            if self.task == 'S':
                new_data.append(img_id)
                new_label.append(raw_label)
            elif self.task == 'T1':
                if raw_label in [0, 1]:
                    new_data.append(img_id)
                    new_label.append(raw_label)
            elif self.task == 'T2':
                if raw_label in [2, 3]:
                    new_data.append(img_id)
                    new_label.append(raw_label - 2)

        self.data = new_data
        self.label = new_label

    # =================================
    # paired augmentation
    # =================================
    def get_aug_img(self, img1, img2):
        # 如果已经是 Tensor，跳过 to_tensor
        if isinstance(img1, torch.Tensor):
            t1 = img1
            t2 = img2
        else:
            t1 = tf.to_tensor(img1)
            t2 = tf.to_tensor(img2)

        # resize
        t1 = tf.resize(t1, [224, 224])
        t2 = tf.resize(t2, [224, 224])

        # 数据增强
        if self.is_train:
            if random.random() > 0.5:
                t1 = tf.hflip(t1)
                t2 = tf.hflip(t2)

            if random.random() > 0.5:
                t1 = tf.vflip(t1)
                t2 = tf.vflip(t2)

            if random.random() > 0.3:
                degree = random.uniform(-10, 10)
                t1 = tf.rotate(t1, degree)
                t2 = tf.rotate(t2, degree)

        # normalize
        t1 = tf.normalize(t1, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        t2 = tf.normalize(t2, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        return t1, t2

    # =================================
    # 读取图像对
    # =================================
    def load_pair(self, img_id):
        fname_C = self.img_map[img_id]["C"]
        fname_G = self.img_map[img_id]["G"]

        path_C = os.path.join(self.img_root, self.dataset, fname_C)
        path_G = os.path.join(self.img_root, self.dataset, fname_G)

        img_C = Image.open(path_C).convert("RGB")
        img_G = Image.open(path_G).convert("RGB")

        img_C, img_G = self.get_aug_img(img_C, img_G)

        return img_C, img_G

    # =================================
    #  few-shot support
    # =================================
    def _build_support_set(self):

        assert self.support_dataset is not None, "support_dataset 必须提供"

        support_C, support_G, support_label = [], [], []

        classes = list(self.support_dataset.class_to_indices.keys())
        classes = classes[:self.args.test_way]   # N-way

        class_map = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            indices = self.support_dataset.class_to_indices[cls]
            chosen = random.sample(indices, self.args.shot)

            for idx in chosen:
                img_id = self.support_dataset.data[idx]
                img_C, img_G = self.support_dataset.load_pair(img_id)

                support_C.append(img_C)
                support_G.append(img_G)
                support_label.append(class_map[cls])

        return (
            torch.stack(support_C),
            torch.stack(support_G),
            torch.tensor(support_label),
            class_map
        )


    # =================================
    # dataset接口
    # =================================
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data[idx]
        label = self.label[idx]

        query_C, query_G = self.load_pair(img_id)

        # 普通模式
        if not self.few_shot:
            return query_C, query_G, label

        if self.fixed_support:
            support_C, support_G, support_label, class_map = self.cached_support
        else:
            support_C, support_G, support_label, class_map = self._build_support_set()

        # few-shot episode
        if label not in class_map:
            return self.__getitem__(random.randint(0, len(self.data)-1))

        query_label = torch.tensor([class_map[label]])

        query_C = query_C.unsqueeze(0)
        query_G = query_G.unsqueeze(0)

        return (
            query_C,
            query_G,
            query_label,
            support_C,
            support_G,
            support_label
        )