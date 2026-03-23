import torch
import os.path as osp
import pprint

path = '/root/autodl-tmp/CSSN_TMI/results/delete-0-out_t1-small-20260316-170816/trlog'

trlog = torch.load(path, map_location='cpu', weights_only=False)
num_epochs = len(trlog['train_loss'])

print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'ValAcc':>6} | {'ValF1':>6} | {'ValBAcc':>7}")
print("-" * 60)

for i in range(num_epochs):
    ep = i + 1

    tr_loss = trlog['train_loss'][i]
    tr_acc  = trlog['train_acc'][i]
    val_acc = trlog['val_acc'][i]
    val_f1  = trlog['val_f1'][i]
    val_bac = trlog['val_bacc'][i]

    print(f"{ep:3d} | "
          f"{tr_loss:7.4f} | "
          f"{tr_acc:6.3f} | "
          f"{val_acc:6.3f} | "
          f"{val_f1:6.3f} | "
          f"{float(val_bac):7.3f}")

print(f"max_acc_epoch: {trlog['max_acc_epoch']}")