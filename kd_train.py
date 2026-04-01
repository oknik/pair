import argparse
import os.path as osp
import os
import copy
from datetime import datetime

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score


from cssn_model import CSSN
from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from utils import pprint, ensure_path, Averager, count_acc, compute_confidence_interval, CircleNum, count_acc_cosine
from tensorboardX import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
import torch.nn.init as init
from pairs.pair_generator import PairGenerator_pcr, PairGenerator_isic
from models.cfl import CFL_ConvBlock
from loss.loss import SoftCELoss, CFLoss, CFLoss_SA
from loss.SDD_DKD import multi_dkd

margin = 0.3

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        # 低通滤波器的不同尺寸，可以通过不同的滤波器尺寸进行处理，大小分别为1x1，2x2，3x3和6x6。
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        # 为每种大小创建一个滤波器（使用 nn.AdaptiveAvgPool2d 进行池化），以适应输入特征图的大小。
        self.relu = nn.ReLU()
        ch =  in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]
        # 将输入特征图的通道分为四部分，每个部分的通道数是 in_channel // 4
    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3) # 获取输入特征图的高度和宽度
        feats = torch.split(feats, self.channel_splits, dim=1) # 将输入的 feats 按照 self.channel_splits 划分成多个部分
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)] # 对每个部分应用 stages[i]，即进行自适应池化后，再通过双线性插值将其上采样至输入图像的原始大小。
        bottle = torch.cat(priors, 1) # 将所有上采样后的部分沿着通道维度拼接。
        
        return self.relu(bottle)

def vit_to_map(x):
    # x: [B, N, C]
    B, N, C = x.shape
    x = x[:, 1:, :]              # 去掉 cls token
    H = W = int((N - 1) ** 0.5)  # 14x14
    x = x.permute(0, 2, 1)       # [B, C, N]
    x = x.reshape(B, C, H, W)    # [B, C, H, W]
    return x

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=dense_predict_network.state_dict()), osp.join(args.save_path, name + '_dense_predict.pth'))
    save_path = '-'.join([args.exp, args.dataset, args.model_type, timestamp])
    args.save_path = osp.join('./results', save_path)
    ensure_path(args.save_path)
    
    if args.dataset == 'in':
        from in_dataset import INDataset as Dataset
        from in_dataset import INTestDataset as TestDataset
        trainset = Dataset(
            img_root='IN',
            dataset='train',
            args=args,
            task='S',
            transform=None,
            fold=0
        )
        pairgenerator = PairGenerator_pcr(trainset, 5, args)
    elif args.dataset == 'out_t1':
        from in_dataset import OUTDataset as Dataset
        from in_dataset import OUTTestDataset as TestDataset
        trainset_t1 = Dataset(
            img_root='OUT',
            dataset='train',
            args=args,
            task='T1',
            transform=None,
            fold=0
        )
        pairgenerator = PairGenerator_pcr(trainset, 5, args)
    elif args.dataset == 'out_t2':
        from in_dataset import OUTDataset as Dataset
        from in_dataset import OUTTestDataset as TestDataset
        trainset_t2 = Dataset(
            img_root='OUT',
            dataset='train',
            args=args,
            task='T2',
            transform=None,
            fold=0
        )
        pairgenerator = PairGenerator_pcr(trainset_t2, 5, args)
    elif args.dataset == 'in_4':
        from student_dataset_4 import StudentDataset as Dataset
        from student_dataset_4 import StudentDataset as TestDataset
        trainset = Dataset(
            img_root='data/IN_4',
            dataset='train',
            task='S',
            args=args,
            fold=0,
            is_train=True
        )
        pairgenerator = PairGenerator_isic(trainset, 5, args)
    else:
        raise ValueError('Non-supported Dataset.')
    sampler = ImbalancedDatasetSampler(trainset)

    train_loader = DataLoader(dataset=trainset, num_workers=8, batch_size=args.batch_size, drop_last=True, shuffle=False, sampler=sampler)
    
    if args.dataset == 'in':
        testset = Dataset(
            img_root='IN',
            dataset='val',
            args=args,
            task='S',
            transform=None,
            fold=0
        )
        valset = TestDataset(trainset, testset, args)
    elif args.dataset == 'out_t1':
        testset = Dataset(
            img_root='OUT',
            dataset='val',
            args=args,
            task='T1',
            transform=None,
            fold=0
        )
        valset = TestDataset(trainset, testset, args)
    elif args.dataset == 'out_t2':
        testset = Dataset(
            img_root='OUT',
            dataset='val',
            args=args,
            task='T2',
            transform=None,
            fold=0
        )
        valset = TestDataset(trainset, testset, args)
    elif args.dataset == 'in_4':
        valset = Dataset(
            img_root='data/IN_4',
            dataset='val',
            task='S',
            args=args,
            is_train=False,
            few_shot=True,
            support_dataset=trainset,
            fixed_support=True
        )
    else:
        valset = TestDataset('val', args)
    val_loader = DataLoader(valset, batch_size=1, num_workers=8)

    t1_model = BackBone(args)
    t1_dense = CSSN(args)
    t2_model = BackBone(args)
    t2_dense = CSSN(args)

    t1_model_ckpt = './results/delete-0-out_t1-small-20260316-170816/max_acc.pth'
    t1_dense_ckpt = './results/delete-0-out_t1-small-20260316-170816/max_acc_dense_predict.pth'
    t2_model_ckpt = './results/delete-0-out_t2-small-20260317-103051/max_acc.pth'
    t2_dense_ckpt = './results/delete-0-out_t2-small-20260317-103051/max_acc_dense_predict.pth'

    t1_model_dict = torch.load(t1_model_ckpt, map_location='cpu')
    t1_dense_dict = torch.load(t1_dense_ckpt, map_location='cpu')
    t2_model_dict = torch.load(t2_model_ckpt, map_location='cpu')
    t2_dense_dict = torch.load(t2_dense_ckpt, map_location='cpu')

    t1_model.load_state_dict(t1_model_dict['params'])
    t1_dense.load_state_dict(t1_dense_dict['params'])
    t2_model.load_state_dict(t2_model_dict['params'])
    t2_dense.load_state_dict(t2_dense_dict['params'])

    t1_model.eval()
    t1_dense.eval()
    t2_model.eval()
    t2_dense.eval()

    t1_model.to(device)
    t1_dense.to(device)
    t2_model.to(device)
    t2_dense.to(device)

    model = BackBone(args)
    dense_predict_network = CSSN(args)
    model = model.to(device)
    dense_predict_network = dense_predict_network.to(device)

    model_dict = model.state_dict()
    if args.init_weights:
        pretrained_dict = torch.load(args.init_weights, map_location='cpu')['teacher']
        pretrained_dict = {k.replace('backbone', 'encoder'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # ---------------- Hook  ----------------
    def forward_hook(module, input, output):
        module.output = output  # 保存特征 maps

    # ----------------  低层特征 ----------------
    # 选择 blocks 的索引做 low-level 特征
    low_layer_idx = 3  # index是3，对应vit_small的第4个block
    t1_model.encoder.blocks[low_layer_idx].register_forward_hook(forward_hook)
    t2_model.encoder.blocks[low_layer_idx].register_forward_hook(forward_hook)
    model.encoder.blocks[low_layer_idx].register_forward_hook(forward_hook)

    t1_feature_dim_SA = t1_model.encoder.embed_dim
    t2_feature_dim_SA = t2_model.encoder.embed_dim
    stu_feature_dim_SA = model.encoder.embed_dim
    t1_low_pass = LowPassModule(in_channel = t1_feature_dim_SA)
    t2_low_pass = LowPassModule(in_channel = t2_feature_dim_SA)

    cfl_blk_SA = CFL_ConvBlock(stu_feature_dim_SA, [t1_feature_dim_SA, t2_feature_dim_SA], 128).to(device)

    # ----------------  高层特征 ----------------
    high_layer_idx = 11  # 最后一个block
    t1_model.encoder.blocks[high_layer_idx].register_forward_hook(forward_hook)
    t2_model.encoder.blocks[high_layer_idx].register_forward_hook(forward_hook)
    model.encoder.blocks[high_layer_idx].register_forward_hook(forward_hook)

    t1_feature_dim = t1_model.encoder.embed_dim
    t2_feature_dim = t2_model.encoder.embed_dim
    stu_feature_dim = model.encoder.embed_dim

    cfl_blk = CFL_ConvBlock(stu_feature_dim, [t1_feature_dim, t2_feature_dim], 128).to(device)

    def count_unfrozen_params(model):
        unfrozen_params = [p for p in model.parameters() if p.requires_grad]
        total_unfrozen = sum(p.numel() for p in unfrozen_params)
        return total_unfrozen / 1e6

    total_params = count_unfrozen_params(model) + count_unfrozen_params(dense_predict_network)
    print(f'Params: {total_params}')
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    dense_predict_network_optim = torch.optim.Adam(dense_predict_network.parameters(), lr=args.lr * args.lr_mul, weight_decay=0.001)
    dense_predict_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dense_predict_network_optim, T_max=100)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion_ce = SoftCELoss(T=1.0)
    criterion_cf = CFLoss(normalized=True)
    criterion_cf_LP = CFLoss_SA(normalized=True)
    BCEloss = torch.nn.BCEWithLogitsLoss()

    global_count = 0
    writer = SummaryWriter(comment=args.save_path)
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    trlog['val_f1'] = []
    trlog['max_f1'] = 0.0
    trlog['max_f1_epoch'] = 0
    trlog['val_bacc'] = []
    trlog['max_bacc'] = 0.0
    trlog['max_bacc_epoch'] = 0
    trlog['val_M'] = []

    cosine_c = 2 ** args.cosine_ratio

    # train
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        dense_predict_network.train()
        tl = Averager()
        ta = Averager()
        for i, batch in enumerate(train_loader, 1):
        
            optimizer.zero_grad()
            dense_predict_network_optim.zero_grad()

            global_count = global_count + 1
            
            if args.dataset in {'isic', '7pt'}:
                data, t, idx = [_.cuda() for _ in batch]
                data_shot, data_query, label = pairgenerator.batch_generator(epoch, data, t, idx)
                data_shot_start = copy.deepcopy(data_shot)
                data_query_start = copy.deepcopy(data_query)
            elif args.dataset in {'in', 'out_t1', 'out_t2'}:
                data, t, data_start= [_.cuda() for _ in batch]
                data_shot_C, data_shot_G, data_query_C, data_query_G, label = pairgenerator.batch_generator(epoch, data, data_start, t)
                data_shot = torch.cat([data_shot_C, data_shot_G], dim=1)   # [B,6,224,224]
                data_query = torch.cat([data_query_C, data_query_G], dim=1) # [B,6,224,224]
                data_shot_start = copy.deepcopy(data_shot)
                data_query_start = copy.deepcopy(data_query)
            elif args.dataset in {'in_4'}:
                data, data_start, t = [_.cuda() for _ in batch]
                data_shot, data_shot_start, data_query, data_query_start, label = pairgenerator.batch_generator(epoch, data, data_start, t)
                data_shot = torch.cat([data_shot, data_shot_start], dim=1)   # [B,6,224,224]
                data_query = torch.cat([data_query, data_query_start], dim=1) # [B,6,224,224]
                data_shot_start = copy.deepcopy(data_shot)
                data_query_start = copy.deepcopy(data_query)
            else:
                data, t, data_start= [_.cuda() for _ in batch]
                data_shot, data_shot_start, data_query, data_query_start, label = pairgenerator.batch_generator(epoch, data, data_start, t)
            
            # Teacher 特征
            with torch.no_grad():
                t1_feat_shot, t1_feat_query = t1_model(data_shot, data_query)
                t2_feat_shot, t2_feat_query = t2_model(data_shot, data_query)

                ft1_SA = t1_model.encoder.blocks[3].output
                ft2_SA = t2_model.encoder.blocks[3].output
                ft1_SA_map = vit_to_map(ft1_SA)
                ft2_SA_map = vit_to_map(ft2_SA)

                ft1 = t1_model.encoder.blocks[11].output
                ft2 = t2_model.encoder.blocks[11].output
                ft1_map = vit_to_map(ft1)
                ft2_map = vit_to_map(ft2)

            # Student 特征
            feat_shot, feat_query = model(data_shot, data_query)
            feat_shot_start, feat_query_start = model(data_shot_start, data_query_start)

            fs_SA = model.encoder.blocks[3].output
            fs_SA_map = vit_to_map(fs_SA)
            fs = model.encoder.blocks[11].output
            fs_map = vit_to_map(fs)

            # 低层
            ft1_LP = t1_low_pass(ft1_SA_map)
            ft2_LP = t2_low_pass(ft2_SA_map)
            ft_LP = [ft1_LP, ft2_LP]
            (hs_LP, ht_LP), (ft_LP_, ft_LP) = cfl_blk_SA(fs_SA_map, ft_LP)
            loss_cf_LP = 10*criterion_cf_LP(hs_LP, ht_LP) #浅层特征的MMD损失,没有重构损失

            # 高层特征
            ft = [ft1_map, ft2_map]
            (hs, ht), (ft_, ft) = cfl_blk(fs_map, ft)
            loss_cf = 10*criterion_cf(hs, ht, ft_, ft) #MMD和重构损失

            alpha = 1
            beta = 8
            temperature = 4
            # loss_ce = multi_dkd(patch_score_s, patch_t_outs, labels, alpha, beta, temperature)

            results, cosine, query_class, support_class = dense_predict_network(feat_query, feat_shot, 'train', feat_query_start, feat_shot_start)
            loss1 = BCEloss(results, label.float())

            cosine = torch.sigmoid(cosine)
            result = torch.sigmoid(results).detach()
            label_cosine_tmp = -1 * torch.log2((1 - cosine_c) * result + cosine_c) + 1
            label_cosine = label * label_cosine_tmp
            label_cosine = label_cosine.detach()
            loss2 = BCEloss(cosine, label_cosine)

            loss_total = (1 - args.cosine_weight) * loss1 + args.cosine_weight * loss2 + loss_cf + loss_cf_LP

            loss_total.backward()

            optimizer.step()
            dense_predict_network_optim.step()
            
            acc = count_acc_cosine(results.data, label, args.cosine_weight)
            writer.add_scalar('data/loss', float(loss_total), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            writer.add_scalar('data/loss_result', float(loss1), global_count)
            writer.add_scalar('data/loss_cosine', float(loss2), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} loss_result={:.4f} loss_cosine={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss_total.item(), loss1.item(), loss2.item(), acc))
            tl.add(loss_total.item())
            ta.add(acc)
        
        lr_scheduler.step()
        dense_predict_network_scheduler.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()
        dense_predict_network.eval()
        vl = Averager()
        va = Averager()
        preds = []
        labels = []
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        print('best f1 epoch {}, best val f1={:.4f}'.format(trlog['max_f1_epoch'], trlog['max_f1']))
        print('best bacc epoch {}, best val bacc={:.4f}'.format(trlog['max_bacc_epoch'], trlog['max_bacc']))
        # test
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if args.dataset in {'isic', '7pt'}:
                    data, t, train_data, _ = [_.cuda() for _ in batch]
                    train_data_start = copy.deepcopy(train_data)
                    data_start = copy.deepcopy(data)
                elif args.dataset in {'in'}:
                    data, t, data_start, train_data, train_label, train_data_start = [_.cuda() for _ in batch]
                    data_query = torch.cat([data, data_start], dim=2)
                    data_shot = torch.cat([train_data, train_data_start], dim=2)
                    data_query_start = copy.deepcopy(data_query)
                    data_shot_start = copy.deepcopy(data_shot)
                elif args.dataset in {'in_4'}:
                    data, data_start, t, train_data, train_data_start, train_label = [_.cuda() for _ in batch]
                    data_shot_C = torch.cat([train_data[:, :3], train_data_start[:, :3]], dim=2)
                    data_shot_G = torch.cat([train_data[:, 3:], train_data_start[:, 3:]], dim=2)
                    data_query_C = torch.cat([data[:, :3], data_start[:, :3]], dim=2)
                    data_query_G = torch.cat([data[:, 3:], data_start[:, 3:]], dim=2)
                    data_shot = torch.cat([data_shot_C, data_shot_G], dim=1)   # [B,6,224,224]
                else:
                    data, t, data_start, train_data, train_label, train_data_start = [_.cuda() for _ in batch]

                data_shot = data_shot.squeeze(0)
                data_shot_start = data_shot_start.squeeze(0)

                data_query = data_query.squeeze(0)
                data_query_start = data_query_start.squeeze(0)
                
                # data_query = torch.repeat_interleave(data, repeats=args.num_classes * args.query_num, dim=0)
                # data_query_start = torch.repeat_interleave(data_start, repeats=args.num_classes * args.query_num, dim=0)
                
                feat_shot, feat_query = model(data_shot, data_query)
                feat_shot_start, feat_query_start = model(data_shot_start, data_query_start)

                results, cosine, _, _ = dense_predict_network(feat_query, feat_shot, 'test', feat_query_start, feat_shot_start)  # Q x S
            
                score = torch.sigmoid(results).detach().cpu().numpy()

                num_segments = len(score) // args.query_num
    
                segment_sums = []
                
                for i in range(num_segments):
                    start_index = i * args.query_num
                    end_index = start_index + args.query_num
                    segment_sum = score[start_index:end_index].sum()
                    segment_sums.append(segment_sum)
                pred = np.argmax(segment_sums)
                preds.append(pred)
                labels.append(int(t.detach().cpu()))
                if t.data != pred:
                    acc = 0
                else:
                    acc = 1
                va.add(acc)
        va = va.item()
        writer.add_scalar('data/val_acc', float(va), epoch)

        preds = np.array(preds)
        labels = np.array(labels)
        confusion_mat = confusion_matrix(labels, preds)
        
        # acc_sk = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat[:, :])
        acc_sk = np.trace(confusion_mat) / np.sum(confusion_mat)
        f1 = f1_score(labels, preds) if args.dataset == 'pcr' or args.dataset == 'cifar' else f1_score(labels, preds, average='macro')
        bacc = np.array([confusion_mat[i, i] / np.sum(confusion_mat[i, :]) for i in range(len(confusion_mat[0]))]).sum() / len(confusion_mat[0])

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')
        if args.dataset == 'pcr' or args.dataset == 'cifar':
            if f1 >= trlog['max_f1']:
                trlog['max_f1'] = f1
                trlog['max_f1_epoch'] = epoch
                save_model('max_f1')
        elif args.dataset == 'isic' or args.dataset == '7pt':
            if bacc >= trlog['max_bacc']:
                trlog['max_bacc'] = bacc
                trlog['max_bacc_epoch'] = epoch
                save_model('max_bacc')

        print(confusion_mat)
        print('epoch {}, val, acc={:.4f}, sk_acc={:.4f}, f1={:.4f}, bacc={:.4f}'.format(epoch, va, acc_sk, f1, bacc))
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_acc'].append(va)
        trlog['val_f1'].append(f1)
        trlog['val_bacc'].append(bacc)
        trlog['val_M'].append(confusion_mat)
        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        save_model('epoch-last')        
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=70)
    parser.add_argument('--way', type=int, default=2)
    parser.add_argument('--test_way', type=int, default=2)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00001) #0.00001
    parser.add_argument('--lr_mul', type=float, default=100)# 100
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='in', choices=['in', 'out_t1', 'out_t2', 'in_4'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--exp', type=str, default='delete')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--query_num', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--init_weights', type=str, default='')

    parser.add_argument('--cosine_weight', type=float, default=0.1)
    parser.add_argument('--cosine_ratio', type=float, default=0.7)
    parser.add_argument('--ppp', type=int, default=7)

    parser.add_argument('--lam', type=float, default=3.0)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--scale', type=float, default=8)

    
    args = parser.parse_args()

    if args.dataset == 'in':
        args.num_classes = 3
        args.fold = 0
    elif args.dataset == 'out_t1':
        args.num_classes = 2
        args.fold = 0
    elif args.dataset == 'out_t2':
        args.num_classes = 2
        args.fold = 0
    elif args.dataset == 'in_4':
        args.num_classes = 4
        args.fold = 0


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_name = args.exp
    if args.fold == -1:
        for i in range(5):
            args.fold = i
            args.exp = f'{exp_name}-{i}'
            main(args)
    else:
        args.exp = f'{exp_name}-{args.fold}'
        main(args)
    
