import argparse
import os.path as osp
import os
import copy
from datetime import datetime

import numpy as np
import torch
import math
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

margin = 0.3

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=dense_predict_network.state_dict()), osp.join(args.save_path, name + '_dense_predict.pth'))
    save_path = '-'.join([args.exp, args.dataset, args.model_type, timestamp])
    args.save_path = osp.join('./results', save_path)
    ensure_path(args.save_path)
    
    # if args.dataset == 'MiniImageNet':
    #     from dataloader.mini_imagenet import MiniImageNet as Dataset
    # elif args.dataset == 'pcr':
    #     from pcr import MiniImageNet as Dataset
    #     from pcr import pcrTest as TestDataset
    #     trainset = Dataset('train', args)
    #     pairgenerator = PairGenerator_pcr(trainset, 5, args)
    # elif args.dataset == 'isic':
    #     from isic_dataset.isic import isic_2017 as Dataset
    #     from isic_dataset.isic import isic_2017_test as TestDataset
    #     trainset = Dataset('train')
    #     pairgenerator = PairGenerator_isic(trainset, 5, args)
    # elif args.dataset == 'cifar':
    #     from cifar import CIFAR as Dataset
    #     from cifar import cifarTest as TestDataset
    #     trainset = Dataset('train', args)
    #     pairgenerator = PairGenerator_pcr(trainset, 5, args)
    # elif args.dataset == '7pt':
    #     from isic_dataset.sevenpt import SevenPT as Dataset
    #     from isic_dataset.sevenpt import SevenPTTest as TestDataset
    #     trainset = Dataset('train')
    #     pairgenerator = PairGenerator_isic(trainset, 5, args)
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
        from out_dataset import OUTDataset as Dataset
        from out_dataset import OUTTestDataset as TestDataset
        trainset = Dataset(
            img_root='tr_val_test/out_tvt',
            split='train',
            task='T1',
            args=args
        )
        pairgenerator = PairGenerator_pcr(trainset, 5, args)
    elif args.dataset == 'out_t2':
        from out_dataset import OUTDataset as Dataset
        from out_dataset import OUTTestDataset as TestDataset
        trainset = Dataset(
            img_root='tr_val_test/out_tvt',
            split='train',
            task='T2',
            args=args
        )
        pairgenerator = PairGenerator_pcr(trainset, 5, args)
    elif args.dataset == 'out_t1_4':
        from teacher_dataset_4 import TeacherDataset as Dataset
        from teacher_dataset_4 import TeacherDataset as TestDataset
        trainset = Dataset(
            img_root='data/OUT_4',
            dataset='train',
            task='T1',
            args=args
        )
        pairgenerator = PairGenerator_pcr(trainset, 5, args)
    elif args.dataset == 'out_t2_4':
        from teacher_dataset_4 import TeacherDataset as Dataset
        from teacher_dataset_4 import TeacherDataset as TestDataset
        trainset = Dataset(
            img_root='data/OUT_4',
            dataset='train',
            task='T2',
            args=args
        )
        pairgenerator = PairGenerator_pcr(trainset, 5, args)
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
            img_root='tr_val_test/out_tvt',
            split='val',
            task='T1',
            args=args
        )
        valset = TestDataset(trainset, testset, args)
    elif args.dataset == 'out_t2':
        testset = Dataset(
            img_root='tr_val_test/out_tvt',
            split='val',
            task='T2',
            args=args
        )
        valset = TestDataset(trainset, testset, args)
    elif args.dataset == 'out_t1_4':
        valset = Dataset(
            img_root='data/OUT_4',
            dataset='val',
            task='T1',
            args=args,
            is_train=False,
            few_shot=True,
            support_dataset=trainset,
            fixed_support=True
        )
    elif args.dataset == 'out_t2_4':
        valset = Dataset(
            img_root='data/OUT_4',
            dataset='val',
            task='T2',
            args=args,
            is_train=False,
            few_shot=True,
            support_dataset=trainset,
            fixed_support=True
        )
    else:
        valset = TestDataset('val', args)
    val_loader = DataLoader(valset, batch_size=1, num_workers=8)

    model = BackBone(args)
    dense_predict_network = CSSN(args)

    model_dict = model.state_dict()
    if args.init_weights:
        pretrained_dict = torch.load(args.init_weights, map_location='cpu')['teacher']
        pretrained_dict = {k.replace('backbone', 'encoder'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        dense_predict_network = dense_predict_network.cuda()

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
                data_shot, data_shot_start, data_query, data_query_start, label = pairgenerator.batch_generator(epoch, data, data_start, t)
                data_shot = torch.cat([data_shot, data_shot_start], dim=1)   # [B,6,224,224]
                data_query = torch.cat([data_query, data_query_start], dim=1) # [B,6,224,224]
                data_shot_start = copy.deepcopy(data_shot)
                data_query_start = copy.deepcopy(data_query)
            elif args.dataset in {'out_t1_4', 'out_t2_4'}:
                data, data_start, t = [_.cuda() for _ in batch]
                data_shot, data_shot_start, data_query, data_query_start, label = pairgenerator.batch_generator(epoch, data, data_start, t)
                data_shot = torch.cat([data_shot, data_shot_start], dim=1)   # [B,6,224,224]
                data_query = torch.cat([data_query, data_query_start], dim=1) # [B,6,224,224]
                data_shot_start = copy.deepcopy(data_shot)
                data_query_start = copy.deepcopy(data_query)
            else:
                data, t, data_start= [_.cuda() for _ in batch]
                data_shot, data_shot_start, data_query, data_query_start, label = pairgenerator.batch_generator(epoch, data, data_start, t)

            feat_shot, feat_query = model(data_shot, data_query)
            feat_shot_start, feat_query_start = model(data_shot_start, data_query_start)
            results, cosine, query_class, support_class = dense_predict_network(feat_query, feat_shot, 'train', feat_query_start, feat_shot_start)
            loss1 = BCEloss(results, label.float())

            cosine = torch.sigmoid(cosine)
            result = torch.sigmoid(results).detach()
            label_cosine_tmp = -1 * torch.log2((1 - cosine_c) * result + cosine_c) + 1
            label_cosine = label * label_cosine_tmp
            label_cosine = label_cosine.detach()
            loss2 = BCEloss(cosine, label_cosine)

            loss_total = (1 - args.cosine_weight) * loss1 + args.cosine_weight * loss2

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
                elif args.dataset in {'in', 'out_t1', 'out_t2'}:
                    data, t, data_start, train_data, train_label, train_data_start = [_.cuda() for _ in batch]
                    data_query = torch.cat([data, data_start], dim=2)
                    data_shot = torch.cat([train_data, train_data_start], dim=2)
                    data_query_start = copy.deepcopy(data_query)
                    data_shot_start = copy.deepcopy(data_shot)
                elif args.dataset in {'out_t1_4', 'out_t2_4'}:
                    data, data_start, t, train_data, train_data_start, train_label = [_.cuda() for _ in batch]
                    data_query = torch.cat([data, data_start], dim=2)
                    data_shot = torch.cat([train_data, train_data_start], dim=2)
                    data_query_start = copy.deepcopy(data_query)
                    data_shot_start = copy.deepcopy(data_shot)
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
                
                for s in range(num_segments):
                    start_index = s * args.query_num
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
    parser.add_argument('--dataset', type=str, default='in', choices=['pcr', 'isic', 'cifar', '7pt', 'in', 'out_t1', 'out_t2', 'out_t1_4', 'out_t2_4'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--exp', type=str, default='delete')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--query_num', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--init_weights', type=str, default='')

    parser.add_argument('--cosine_weight', type=float, default=0.01)
    parser.add_argument('--cosine_ratio', type=float, default=0.7)
    parser.add_argument('--ppp', type=int, default=7)

    parser.add_argument('--lam', type=float, default=3.0)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--scale', type=float, default=8)

    
    args = parser.parse_args()

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
    elif args.dataset == 'in':
        args.num_classes = 3
        args.fold = 0
    elif args.dataset == 'out_t1' or args.dataset == 'out_t2' or args.dataset == 'out_t1_4' or args.dataset == 'out_t2_4':
        args.num_classes = 2
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
    
