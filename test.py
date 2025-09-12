import argparse
import os.path as osp
import os
import copy
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from sklearn.metrics import confusion_matrix, f1_score
from thop import profile

from cssn_model import CSSN
from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

def main(args):
    
    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'pcr':
        from pcr import MiniImageNet as Dataset
        from pcr import pcrTest as TestDataset
        # from dataloader.pcr import MiniImageNet as Dataset
        # from dataloader.pcr import pcrTest as TestDataset
    elif args.dataset == 'isic':
        from isic_dataset.isic import isic_2017 as Dataset
        from isic_dataset.isic import isic_2017_test as TestDataset
    elif args.dataset == 'cifar':
        from cifar import CIFAR as Dataset
        from cifar import cifarTest as TestDataset
    elif args.dataset == '7pt':
        from isic_dataset.sevenpt import SevenPT as Dataset
        from isic_dataset.sevenpt import SevenPTTest as TestDataset
    else:
        raise ValueError('Non-supported Dataset.')
    
    # 模型初始化
    model = BackBone(args)
    dense_predict_network = CSSN(args)

    model.load_state_dict(torch.load(f'./results/{args.exp}-{args.dataset}-small/max_{args.metric}.pth')['params'])
    dense_predict_network.load_state_dict(torch.load(f'./results/{args.exp}-{args.dataset}-small/max_{args.metric}_dense_predict.pth')['params'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0])
        dense_predict_network = dense_predict_network.cuda()
    
    input_sample = torch.randn(1, 3, 224, 224).cuda()
    input_sample2 = torch.randn(1, 197, 384).cuda()
    flops1, params1 = profile(model, inputs=(input_sample, input_sample))
    flops2, params2 = profile(dense_predict_network, inputs=(input_sample2, input_sample2, 'test', input_sample2, input_sample2))
    print(f'Total FLOPs: {flops1 + flops2}')
    print(f'Total params: {params1 + params2}')

    model.eval()
    dense_predict_network.eval()
    # test
    EPOCHS = 5
    Tacc = 0
    Tprecision = 0
    Trecall = 0
    Tf1 = 0
    Tbacc = 0
    Tspecificity = 0
    Tsensitivity = 0
    Ttime = 0.0
    Tsacc = [0] * 5
    with torch.no_grad():
        for e in range(EPOCHS):
            labels = []
            preds = []
            valset = TestDataset('test', args)
            val_loader = DataLoader(dataset=valset, batch_size=1, num_workers=8, pin_memory=True)
            total_time = 0.0
            for i, batch in enumerate(val_loader, 1):
                if args.dataset in {'isic', '7pt'}:
                    data, t, train_data, _ = [_.cuda() for _ in batch]
                    train_data_start = copy.deepcopy(train_data)
                    data_start = copy.deepcopy(data)
                else:
                    data, t, data_start, train_data, train_label, train_data_start= [_.cuda() for _ in batch]

                data_shot = train_data.squeeze(0)
                data_shot_start = train_data_start.squeeze(0)

                data_query = torch.repeat_interleave(data, repeats=args.num_classes * args.query_num, dim=0)
                data_query_start = torch.repeat_interleave(data_start, repeats=args.num_classes * args.query_num, dim=0)

                start_time = time.perf_counter()
                feat_shot, feat_query = model(data_shot, data_query)
                feat_shot_start, feat_query_start = model(data_shot_start, data_query_start)
                results, cosine, _, _ = dense_predict_network(feat_query, feat_shot, 'test', feat_query_start, feat_shot_start)  # Q x S
                end_time = time.perf_counter()
                total_time += end_time - start_time

                results = torch.sigmoid(results)
                cosine = torch.sigmoid(cosine)
                score = results.detach().cpu().numpy()

                num_segments = len(score) // args.query_num
    
                segment_sums = []
                
                for i in range(num_segments):
                    start_index = i * args.query_num
                    end_index = start_index + args.query_num
                    segment_sum = score[start_index:end_index].sum()
                    segment_sums.append(segment_sum)
                pred = np.argmax(segment_sums)
                labels.append(int(t.detach().cpu()))
                preds.append(pred)
            
            preds = np.array(preds)
            labels = np.array(labels)
            confusion_mat = confusion_matrix(labels, preds)
            avg_time = total_time / len(preds)
        
            acc = np.array([confusion_mat[i, i] for i in range(len(confusion_mat[0]))]).sum() / np.sum(confusion_mat[:, :])
            f1 = f1_score(labels, preds) if args.dataset == 'pcr' or args.dataset == 'cifar' else f1_score(labels, preds, average='macro')
            bacc = np.array([confusion_mat[i, i] / np.sum(confusion_mat[i, :]) for i in range(len(confusion_mat[0]))]).sum() / len(confusion_mat[0])
            print('acc:', acc)
            print('f1:', f1)
            print('bacc:', bacc)
            print(confusion_mat)
            
            precision = confusion_mat[1, 1] / np.sum(confusion_mat[:, 1])
            sensitivity = confusion_mat[1, 1] / np.sum(confusion_mat[1, :])
            recall = sensitivity
            specificity = confusion_mat[0, 0] / np.sum(confusion_mat[0, :])

            Tacc += acc
            Tprecision += precision
            Trecall += recall
            Tf1 += f1
            Tbacc += bacc
            Tspecificity += specificity
            Tsensitivity += sensitivity
            Ttime += avg_time
    r = {
        'acc': Tacc / 5.0,
        'precision': Tprecision / 5.0,
        'recall': Trecall / 5.0,
        'f1': Tf1 / 5.0,
        'bacc': Tbacc / 5.0,
        'specifisity': Tspecificity / 5.0,
        'sensitivity': Tsensitivity / 5.0,
        'times': Ttime / 5.0
        
    }
    print(r)
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--way', type=int, default=2)
    parser.add_argument('--test_way', type=int, default=2)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00001) #0.00001
    parser.add_argument('--lr_mul', type=float, default=100)# 100
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='pcr', choices=['pcr', 'isic', 'cifar', '7pt'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--exp', type=str, default='delete')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--query_num', type=int, default=10)
    parser.add_argument('--metric', type=str, default='acc')
    
    parser.add_argument('--fold', type=int, default=-1)
    
    parser.add_argument('--init_weights', type=str, default='')
    parser.add_argument('--cosine_weight', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--ppp', type=int, default=7)
    parser.add_argument('--lam', type=float, default=3.0)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--scale', type=float, default=8)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dataset == 'pcr':
        args.num_classes = 2
    elif args.dataset == 'cifar':
        args.num_classes = 2
        args.fold = 0
    elif args.dataset == 'isic':
        args.num_classes = 3
        args.fold = 0
    elif args.dataset == '7pt':
        args.num_classes = 5
        args.fold = 0

    exp_name = args.exp
    if args.fold == -1:
        r = []
        for i in range(5):
            args.fold = i
            args.exp = f'{exp_name}-{i}'
            r.append(main(args))
        Tacc = 0
        Tprecision = 0
        Trecall = 0
        Tf1 = 0
        Tbacc = 0
        Tspecificity = 0
        Tsensitivity = 0
        Ttime = 0
        
        for i in range(5):
            Tacc += r[i]['acc']
            Tprecision += r[i]['precision']
            Trecall += r[i]['recall']
            Tf1 += r[i]['f1']
            Tbacc += r[i]['bacc']
            Tspecificity += r[i]['specifisity']
            Tsensitivity += r[i]['sensitivity']
            Ttime += r[i]['times']
        print('5折交叉验证结果')
        print(f'acc:{Tacc / 5.0:.4f}')
        print(f'precision:{Tprecision / 5.0:.4f}')
        print(f'recall:{Trecall / 5.0:.4f}')
        print(f'f1:{Tf1 / 5.0:.4f}')
        print(f'bacc:{Tbacc / 5.0:.4f}')
        print(f'specifisity:{Tspecificity / 5.0:.4f}')
        print(f'sensitivity:{Tsensitivity / 5.0:.4f}')
        print(f'time:{Ttime / 5.0:.4f}')
    else:
        args.exp = f'{exp_name}-{args.fold}'
        main(args)