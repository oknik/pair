import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cosine_Sim(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.layer1 = Mlp(in_features=384, hidden_features=96, out_features=384)

    def forward(self, x, y):
        x = self.layer1(x)
        y = self.layer1(y)

        cosine = F.cosine_similarity(x, y)

        return cosine, x, y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class CSSN(nn.Module):
    def __init__(self, args, in_dim=384):
        super(CSSN, self).__init__()
        self.args = args
        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim/4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)

        self.fc3 = Mlp(in_features=196, hidden_features=128, out_features=196)
        self.fc_norm3 = nn.LayerNorm(196)

        self.fc4 = Mlp(in_features=384, hidden_features=96, out_features=384)
        self.fc_norm4 = nn.LayerNorm(384)
        
        self.pool1 = nn.AvgPool2d(2, 2, 0)
        self.fc2 = Mlp(in_features=98**2, hidden_features=256, out_features=1)

        

        self.cosine_sim = Cosine_Sim(args)


    def forward(self, feat_query, feat_shot, type, feat_query_start, feat_shot_start):
        _, n, c = feat_query.size()

        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query  # Q x n x C
        feat_shot  = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot  # KS x n x C
        feat_query_start  = self.fc1(torch.mean(feat_query_start, dim=1, keepdim=True)) + feat_query_start  # KS x n x C
        feat_shot_start  = self.fc1(torch.mean(feat_shot_start, dim=1, keepdim=True)) + feat_shot_start  # KS x n x C
        feat_query = self.fc_norm1(feat_query)
        feat_shot  = self.fc_norm1(feat_shot)
        feat_query_start  = self.fc_norm1(feat_query_start)
        feat_shot_start  = self.fc_norm1(feat_shot_start)

        query_class = feat_query[:, 0, :].unsqueeze(1)  # Q x 1 x C
        query_image = feat_query[:, 1:, :]  # Q x L x C
        query_image_start = feat_query_start[:, 1:, :]

        support_class = feat_shot[:, 0, :].unsqueeze(1)  # KS x 1 x C
        support_image = feat_shot[:, 1:, :]  # KS x L x C
        support_image_start = feat_shot_start[:, 1:, :]
        
        # treatment fusion
        query_image_start = self.fc_norm3(self.fc3(query_image_start.transpose(-1, -2))).transpose(-1, -2)
        support_image_start = self.fc_norm3(self.fc3(support_image_start.transpose(-1, -2))).transpose(-1, -2)
        query_image = self.fc_norm4(self.fc4(query_image * query_image_start))
        support_image = self.fc_norm4(self.fc4(support_image * support_image_start))

        feat_query = query_image + self.args.lam * query_class  # Q x L x C
        feat_shot = support_image + self.args.lam * support_class  # KS x L x C

        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)
        out = torch.matmul(feat_query, feat_shot.transpose(1, 2))

        out = self.pool1(out)

        out = out.flatten(1)
        # pooling
        out = self.fc2(out.pow(2)).squeeze(1)
        cosine, _, _ = self.cosine_sim(query_class.squeeze(1), support_class.squeeze(1))

        if type == 'train':
            return out, cosine, query_class, support_class
        elif type == 'test':
            return out, cosine, None, None

