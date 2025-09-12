import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
import numpy as np
import json
import torch
import random
import os
import re

from configs.config_path import PATHS

DATASET_DIR = PATHS['T2_path']
CycleGAN = False
XMGAN = False

class MiniImageNet(Dataset):
    """ Usage:
    """

    def __init__(self, setname, args):
        with open(PATHS['T2_json'], 'r') as inf:
            data_dict = json.load(inf)

        if setname == 'val': setname = 'test'
        data_list = data_dict[setname][str(args.fold)]

        data = list()
        label = list()
        data_start = list()
        for image in os.listdir(DATASET_DIR):
            head = head = image[:image.find('_')]
            if (head in data_list) and 'end' in image:
            # if (head in data_list or 'B' in head) and 'end' in image:
                start_filename = re.sub('end', 'start', image)
                data.append(osp.join(DATASET_DIR, image))
                data_start.append(osp.join(DATASET_DIR, start_filename))
                _label = image.split('_')[1]
                label.append(int(_label))

        self.data = data
        self.data_start = data_start
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname

        image_size = 224
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46, 0.46, 0.46], std=[0.1582, 0.1582, 0.1582]),
        ])
        self.transform_val_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46, 0.46, 0.46], std=[0.1582, 0.1582, 0.1582])
        ])
    
    def get_aug_img(self, img1, img2):
        img1 = transforms.ToPILImage()(img1)
        img2 = transforms.ToPILImage()(img2)


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
        tmp_r = random.random()
        if tmp_r < 0.2:
            bright = random.uniform(1, 1.5)
            img1 = tf.adjust_brightness(img1, bright)
            img2 = tf.adjust_brightness(img2, bright)
        elif tmp_r < 0.4:
            bright = random.uniform(0.5, 1)
            img1 = tf.adjust_brightness(img1, bright)
            img2 = tf.adjust_brightness(img2, bright)
        tmp_r = random.random()
        if tmp_r < 0.2:
            bright = random.uniform(1, 1.5)
            img1 = tf.adjust_contrast(img1, bright)
            img2 = tf.adjust_contrast(img2, bright)
        elif tmp_r < 0.4:
            bright = random.uniform(0.5, 1)
            img1 = tf.adjust_contrast(img1, bright)
            img2 = tf.adjust_contrast(img2, bright)
        tmp_r = random.random()
        if tmp_r < 0.2:
            bright = random.uniform(1, 1.5)
            img1 = tf.adjust_saturation(img1, bright)
            img2 = tf.adjust_saturation(img2, bright)
        elif tmp_r < 0.4:
            bright = random.uniform(0.5, 1)
            img1 = tf.adjust_saturation(img1, bright)
            img2 = tf.adjust_saturation(img2, bright)
        
        img1 = tf.to_tensor(img1)
        img2 = tf.to_tensor(img2)
        img1 = tf.normalize(img1, mean=[0.46, 0.46, 0.46], std=[0.1582, 0.1582, 0.1582])
        img2 = tf.normalize(img2, mean=[0.46, 0.46, 0.46], std=[0.1582, 0.1582, 0.1582])
        return img1, img2

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.label

    def __getitem__(self, i):
        path, label, start_path = self.data[i], self.label[i], self.data_start[i]
        def img_pre(p):
            im = Image.open(p).convert('RGB')
            im = tf.resize(im, [224, 224])
            im = tf.to_tensor(im)
            return im

        if self.setname == 'train':
            image = img_pre(path)
            image_start = img_pre(start_path)


        return image, label, image_start
    
class pcrTest(Dataset):
    """ Usage:
    """

    def __init__(self, setname, args):

        with open(PATHS['T2_json'], 'r') as inf:
            data_dict = json.load(inf)

        if setname == 'val': setname = 'test'
        data_list = data_dict[setname][str(args.fold)]
        train_data_list = data_dict['train'][str(args.fold)]

        data = list()
        data_start = list()
        train_data = list()
        train_data_start = list()
        train_label = list()
        label = list()
        for image in os.listdir(DATASET_DIR):
            head = head = image[:image.find('_')]
            if head in data_list and 'end' in image:
                if head == 'A23':
                    continue
                start_file = re.sub('end', 'start', image)
                data.append(osp.join(DATASET_DIR, image))
                data_start.append(osp.join(DATASET_DIR, start_file))
                _label = image.split('_')[1]
                label.append(int(_label))
            elif head in train_data_list and 'end' in image:
                if head == 'A23':
                    continue
                start_file = re.sub('end', 'start', image)
                train_data.append(osp.join(DATASET_DIR, image))
                train_data_start.append(osp.join(DATASET_DIR, start_file))
                _label = image.split('_')[1]
                train_label.append(int(_label))

        self.data = data
        self.data_start = data_start
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname
        train_label = np.array(train_label)
        train_label_zero = np.squeeze(np.argwhere(train_label == 0))
        train_label_one = np.squeeze(np.argwhere(train_label == 1))

        train_sample_zero = np.random.choice(train_label_zero, size=args.query_num, replace=False)
        train_sample_one = np.random.choice(train_label_one, size=args.query_num, replace=False)
        self.train_data = []
        self.train_label = []
        self.train_data_start = []
        for item in train_sample_zero:
            self.train_data.append(train_data[item])
            self.train_data_start.append(train_data_start[item])
            self.train_label.append(0)
        for item in train_sample_one:
            self.train_data.append(train_data[item])
            self.train_data_start.append(train_data_start[item])
            self.train_label.append(1)
        


        image_size = 224
        self.transform_val_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46, 0.46, 0.46], std=[0.1582, 0.1582, 0.1582])
        ])
        images = []
        for item in self.train_data:
            image = self.transform_val_test(Image.open(item).convert('RGB'))
            images.append(image.unsqueeze(0))
        images_start = []
        for item in self.train_data_start:
            image = self.transform_val_test(Image.open(item).convert('RGB'))
            images_start.append(image.unsqueeze(0))
        self.train_images = torch.cat(images, dim=0)
        self.train_images_start = torch.cat(images_start, dim=0)
        self.train_label = np.array(self.train_label)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label, path_start = self.data[i], self.label[i], self.data_start[i]
        image = self.transform_val_test(Image.open(path).convert('RGB'))
        image_start = self.transform_val_test(Image.open(path_start).convert('RGB'))
        return image, label, image_start, self.train_images, self.train_label, self.train_images_start

