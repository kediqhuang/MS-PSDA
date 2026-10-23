import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss_funcs import TransferLoss, SupConLoss
from models import PrototypicalNet



class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return x

    def get_parameters(self):
        return [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]


class MixedSourceClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, num_classes)

    def forward(self, feature):
        f1 = self.fc1(feature)
        f2 = self.fc2(f1)
        return self.fc3(f2)

    def predict(self, feature):
        with torch.no_grad():
            logits = F.softmax(self.forward(feature), dim=1)
            y_preds = torch.argmax(logits, axis=1)
        return y_preds

    def get_parameters(self):
        return [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
            {"params": self.fc3.parameters(), "lr_mult": 1},
        ]


class CrossDomainPrototypicalNet(PrototypicalNet):
    def __init__(self,
                 input_dim,
                 num_classes=14,
                 num_tgt_clusters=15,
                 src_momentum=0.5,
                 tgt_momentum=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.src_momentum = src_momentum
        self.tgt_momentum = tgt_momentum
        self.input_dim = input_dim

        self.num_tgt_clusters = num_tgt_clusters
        self.num_src_clusters = 0

        self.register_buffer('tgt_cluster_centers', torch.zeros(num_tgt_clusters, input_dim))

    def update(self, src_feat, src_cluster, tgt_feat, tgt_cluster):
        self.update_source_cluster_centers(src_feat, src_cluster)
        new_tgt_centers = self.update_target_cluster_centers(tgt_feat, tgt_cluster)
        # return self.generate_target_cluster_sim()

    @torch.no_grad()
    def generate_target_cluster_sim(self):
        return super().generate_target_cluster_sim(self.tgt_cluster_centers)

    @torch.no_grad()
    def initialize_cluster_labels(self, num_clusters_list):
        tmp_labels = np.repeat(range(len(num_clusters_list)), num_clusters_list)
        new_labels = torch.from_numpy(tmp_labels).to(self.tgt_cluster_centers.device)
        self.num_src_clusters = len(new_labels)
        self.register_buffer('src_cluster_labels', torch.zeros(self.num_src_clusters, dtype=torch.long))
        self.register_buffer('src_cluster_centers', torch.zeros(self.num_src_clusters, self.input_dim))
        self.src_cluster_labels.copy_(new_labels)


class IntraSourcePrototypicalNet(PrototypicalNet):
    def __init__(self,
                 input_dim=64,
                 num_classes=3,
                 num_src_clusters=15,
                 num_tgt_clusters=15,
                 src_momentum=0.5,
                 tgt_momentum=0.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_src_clusters = num_src_clusters
        self.num_tgt_clusters = num_tgt_clusters
        self.src_momentum = src_momentum
        self.tgt_momentum = tgt_momentum

        # 使用 register_buffer 确保设备自动跟随模型
        self.register_buffer('src_cluster_labels', torch.zeros(num_src_clusters, dtype=torch.long))
        self.register_buffer('src_cluster_centers', torch.zeros(num_src_clusters, input_dim))
        self.register_buffer('tgt_cluster_centers', torch.zeros(num_tgt_clusters, input_dim))

        # bottomneck
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, x):
        return self.extractor(x)

    def update_all(self, src_feat, src_cluster, tgt_feat, tgt_cluster):
        src_feat = self.extractor(src_feat)
        tgt_feat = self.extractor(tgt_feat)
        self.update_source_cluster_centers(src_feat, src_cluster)
        new_tgt_centers = self.update_target_cluster_centers(tgt_feat, tgt_cluster)
        tgt_cluster_label = self.generate_target_cluster_labels()
        return tgt_cluster_label[tgt_cluster], self.generate_target_cluster_sim(new_tgt_centers)

    def update_target(self, tgt_feat, tgt_cluster):
        tgt_feat = self.extractor(tgt_feat)
        new_tgt_centers = self.update_target_cluster_centers(tgt_feat, tgt_cluster)
        tgt_cluster_label = self.generate_target_cluster_labels()
        return tgt_cluster_label[tgt_cluster], self.generate_target_cluster_sim(new_tgt_centers)

    @torch.no_grad()
    def predict_label(self, tgt_feat):
        tgt_feat = self.extractor(tgt_feat)
        return super().predict_label(tgt_feat)

    @torch.no_grad()
    def predict_prob(self, tgt_feat):
        tgt_feat = self.extractor(tgt_feat)
        return super().predict_prob(tgt_feat)

    @torch.no_grad()
    def initialize_cluster_centers(self, feature, cluster):
        feature = self.extractor(feature)
        super().initialize_cluster_centers(feature, cluster)

    def get_parameters(self):
        return [{"params": self.extractor.parameters(), "lr_mult": 1}]


class MSPSDA(nn.Module):
    def __init__(self,
                 input_dim=310,
                 num_classes=3,
                 max_iter=1000,
                 last_iter=0,
                 transfer_loss_type="dann",
                 num_src_clusters=15,
                 num_tgt_clusters=15,
                 num_sources=14,
                 src_momentum=0.5,
                 tgt_momentum=0.1,
                 ablation="main",
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_sources = num_sources

        self.feature_extractor = FeatureExtractor(input_dim=input_dim, hidden_1=64, hidden_2=64)

        self.cross_domain_cluster = CrossDomainPrototypicalNet(input_dim=64,
                                                       num_classes=num_sources,
                                                       num_tgt_clusters=num_tgt_clusters,
                                                       src_momentum=src_momentum,
                                                       tgt_momentum=tgt_momentum)

        self.mixed_source_classifier = MixedSourceClassifier(input_dim=64, num_classes=num_classes)

        self.multi_intra_source_clusters = nn.ModuleList([
            IntraSourcePrototypicalNet(
                input_dim=64,
                num_classes=num_classes,
                num_src_clusters=num_src_clusters,
                num_tgt_clusters=num_tgt_clusters,
                src_momentum=src_momentum,
                tgt_momentum=tgt_momentum,
            ) for _ in range(num_sources)
        ])

        self.cls_loss = nn.CrossEntropyLoss()
        self.consis_loss = nn.CrossEntropyLoss()
        self.clu_loss = SupConLoss(temperature=1)

        self.domain_adapter = TransferLoss(
            loss_type=transfer_loss_type,
            max_iter=max_iter,
            num_class=num_classes,
            **kwargs
        )

        self.ablation = ablation

    def mixed_source_forward(self, src_feat, tgt_feat, src_label):
        src_logit = self.mixed_source_classifier(src_feat)
        loss_cls = self.cls_loss(src_logit, src_label)
        loss_adv = self.domain_adapter(src_feat, tgt_feat)
        return loss_cls, loss_adv

    def multi_source_forward(self, feat, cluster, weights=None):
        clu_loss = self.clu_loss(
            feat,
            cluster.squeeze(0),
            weights
        )
        return clu_loss

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster, cross_src_cluster, src_idx, step="classifier"):
        # N B F
        srcs = srcs.permute(1, 0, 2)
        _, _, feature_dim = srcs.shape
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        # 1. 特征提取 --> 特征  mmd损失
        # B F
        tgt_feat = self.feature_extractor(tgt)
        src_feat = self.feature_extractor(srcs[src_idx])
        src_cluster = src_clusters[src_idx]

        # 2. 混合源分支 --> 特征重新提取    分类损失  域适应损失
        src_label = src_labels[src_idx]
        loss_cls, loss_adv = self.mixed_source_forward(src_feat, tgt_feat, src_label)

        # 2.5. 跨源域分支 -->
        srcs_merged = srcs.reshape(-1, feature_dim)
        src_feat_merged = self.feature_extractor(srcs_merged)
        cross_src_clu_loss = self.clu_loss(src_feat_merged, cross_src_cluster.squeeze(0))
        self.cross_domain_cluster.update(src_feat_merged, cross_src_cluster, tgt_feat, tgt_cluster)
        # cross_prob = self.cross_domain_cluster.predict_prob(src_feat)
        cross_prob = self.cross_domain_cluster.predict_prob(tgt_feat)

        # 3. 多源聚类分支-特征提取 --> 多源特征    源域聚类损失  目标域聚类损失
        src_cluster_feat = self.multi_intra_source_clusters[src_idx](src_feat)
        src_clu_loss = self.multi_source_forward(src_cluster_feat, src_cluster)

        tgt_clu_loss = 0.
        for i in range(len(self.multi_intra_source_clusters)):
            # batch size
            weights = cross_prob[:, i]
            tgt_cluster_feat = self.multi_intra_source_clusters[i](tgt_feat)
            tgt_clu_loss += self.multi_source_forward(tgt_cluster_feat, tgt_cluster, weights)
        #     tgt_clu_loss += self.multi_source_forward(tgt_cluster_feat, tgt_cluster)
        # tgt_clu_loss = tgt_clu_loss/len(self.multi_intra_source_clusters)

        # 4. 多源聚类分支-聚类标签 --> 更新聚类中心     dice聚类相似度（目标域聚类中心和标签）一致性损失    混合分支和多分支一致性损失
        tgt_preds = []
        tgt_cls_sims = []
        for i in range(len(self.multi_intra_source_clusters)):
            if i == src_idx:
                tgt_pred, tgt_cls_sim = self.multi_intra_source_clusters[src_idx].update_all(src_feat, src_cluster, tgt_feat, tgt_cluster)
                tgt_cls_sims.append(tgt_cls_sim)
            else:
                tgt_pred, tgt_cls_sim = self.multi_intra_source_clusters[i].update_target(tgt_feat, tgt_cluster)
                tgt_cls_sims.append(tgt_cls_sim)
            tgt_preds.append(tgt_pred)

        dice_loss = 0.
        for i in range(len(tgt_cls_sims)):
            if i != src_idx:
                dice_loss += self.DICELoss(tgt_cls_sims[i], tgt_cls_sims[src_idx], cross_prob)
        dice_loss /= len(tgt_cls_sims)

        tgt_logit = self.mixed_source_classifier(tgt_feat)
        loss_consis=0.
        for i in range(len(tgt_preds)):
            # batchsize
            weights = cross_prob[:, i]
            log_probs = torch.nn.functional.log_softmax(tgt_logit, dim=1)
            loss_per_sample = -log_probs[range(tgt_logit.size(0)), tgt_preds[i].to(tgt_logit.device)]
            weighted_loss = (weights * loss_per_sample).mean()
            loss_consis += weighted_loss
            # loss_consis += self.consis_loss(tgt_logit, tgt_preds[i].to(tgt_logit.device))
        # loss_consis = loss_consis/len(tgt_preds)

        if step == "classifier":
            return loss_cls, cross_src_clu_loss, loss_adv, loss_consis
        else:
            return src_clu_loss, tgt_clu_loss, dice_loss

    def predict(self, data, branch=None, mode=None, sv_mode=None):
        if branch is None:
            branch = ["multi-source"]
        if mode is None:
            mode = ["soft-voting"]
        if sv_mode is None:
            sv_mode = ["average"]
        outputs = []
        for b in branch:
            if b == "mixed-source":
                outputs.extend(self.predict_mixed_source(data))
            if b == "multi-source":
                for m in mode:
                    if m == "soft-voting":
                        outputs.extend(self.predict_multi_source_softvoting(data, mode=sv_mode))
                    if m == "hard-voting":
                        outputs.extend(self.predict_multi_source_hardvoting(data))
        return outputs

    def predict_multi_source_hardvoting(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            all_preds = []
            for net in self.multi_intra_source_clusters:
                pred = net.predict_label(feat)
                all_preds.append(pred.unsqueeze(1))
            all_preds = torch.cat(all_preds, dim=1)
            final_pred, _ = torch.mode(all_preds, dim=1)
        return [final_pred]

    def predict_multi_source_softvoting(self, data, mode):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)  # [batch_size, 64]
            all_probs = []
            for net in self.multi_intra_source_clusters:
                prob = net.predict_prob(feat)  # [batch_size, num_classes]
                all_probs.append(prob.unsqueeze(0))  # 增加一维表示第 i 个网络
            # all_probs: [num_sources, batch_size, num_classes]
            all_probs = torch.cat(all_probs, dim=0)
            final_preds = []
            for m in mode:
                if "average" == m:
                    _prob = torch.mean(all_probs, dim=0)  # [batch_size, num_classes]
                    final_pred = torch.argmax(_prob, dim=1)  # [batch_size]
                    final_preds.append(final_pred)
                if "weighted" == m:
                    # cross_prob: [batch_size, num_sources]
                    cross_prob = self.cross_domain_cluster.predict_prob(feat)
                    _prob = torch.einsum('n b c, b n -> b c', all_probs, cross_prob)
                    final_pred = torch.argmax(_prob, dim=1)  # [batch_size]
                    final_preds.append(final_pred)
            return final_preds



    def predict_mixed_source(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            pred = self.mixed_source_classifier.predict(feat)
        return [pred]

    def get_step1_parameters(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.mixed_source_classifier.get_parameters(),
            {"params": self.domain_adapter.loss_func.domain_classifier.parameters(), "lr_mult": 1}
        ]
        return params

    def get_step2_parameters(self):
        params = []
        for net in self.multi_intra_source_clusters:
            params.extend(net.get_parameters())
        return params

    def get_parameter(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.mixed_source_classifier.get_parameters(),
        ]
        for net in self.multi_intra_source_clusters:
            params.extend(net.get_parameters())
        params.append(
            {"params": self.domain_adapter.loss_func.domain_classifier.parameters(), "lr_mult": 1}
        )
        return params

    @torch.no_grad()
    def initialize_cluster(self, srcs, src_clusters, src_labels, src_num_clusters_list, cross_src_clusters):
        feats = [self.feature_extractor(src) for src in srcs]
        feats_cat = torch.cat(feats, dim=0)
        self.cross_domain_cluster.initialize_cluster_labels(src_num_clusters_list)
        self.cross_domain_cluster.initialize_cluster_centers(feats_cat, cross_src_clusters)
        self.cross_domain_cluster.src_cluster_labels = self.cross_domain_cluster.src_cluster_labels.to(feats[0].device)
        self.cross_domain_cluster.src_cluster_centers = self.cross_domain_cluster.src_cluster_centers.to(feats[0].device)
        self.cross_domain_cluster.tgt_cluster_centers = self.cross_domain_cluster.tgt_cluster_centers.to(feats[0].device)
        for i, net in enumerate(self.multi_intra_source_clusters):
            net.initialize_cluster_labels(src_labels[i], src_clusters[i])
            net.initialize_cluster_centers(feats[i], src_clusters[i])
            net.src_cluster_labels = net.src_cluster_labels.to(feats[i].device)
            net.src_cluster_centers = net.src_cluster_centers.to(feats[i].device)
            net.tgt_cluster_centers = net.tgt_cluster_centers.to(feats[i].device)

    def get_state(self):
        multi_intra_source_clusters_extractor = [net.state_dict() for net in self.multi_intra_source_clusters]
        return {
            "model": self.state_dict(),
            "multi_intra_source_clusters_extractor": multi_intra_source_clusters_extractor,
            "multi_intra_source_clusters": {
                "src_cluster_labels": [net.src_cluster_labels.clone().detach().cpu() for net in self.multi_intra_source_clusters],
                "src_cluster_centers": [net.src_cluster_centers.clone().detach().cpu() for net in self.multi_intra_source_clusters],
                "tgt_cluster_centers": [net.tgt_cluster_centers.clone().detach().cpu() for net in self.multi_intra_source_clusters],
            },
            "cross_source_cluster": {
                "src_cluster_labels": self.cross_domain_cluster.src_cluster_labels.clone().detach().cpu(),
                "src_cluster_centers": self.cross_domain_cluster.src_cluster_centers.clone().detach().cpu(),
                "tgt_cluster_centers": self.cross_domain_cluster.tgt_cluster_centers.clone().detach().cpu(),
            }
        }

    def load_state(self, state):
        self.load_state_dict(state["model"], strict=False)
        for i, net in enumerate(self.multi_intra_source_clusters):
            net.load_state_dict(state["multi_intra_source_clusters_extractor"][i])
            net.src_cluster_labels = state["multi_intra_source_clusters"]["src_cluster_labels"][i].to(net.src_cluster_labels.device)
            net.src_cluster_centers = state["multi_intra_source_clusters"]["src_cluster_centers"][i].to(net.src_cluster_centers.device)
            net.tgt_cluster_centers = state["multi_intra_source_clusters"]["tgt_cluster_centers"][i].to(net.tgt_cluster_centers.device)
        self.cross_domain_cluster.src_cluster_labels = state["cross_source_cluster"]["src_cluster_labels"]
        self.cross_domain_cluster.src_cluster_centers = state["cross_source_cluster"]["src_cluster_centers"]
        self.cross_domain_cluster.tgt_cluster_centers = state["cross_source_cluster"]["tgt_cluster_centers"]

    def MMDLoss(self, X, Y):
        delta = X - Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss

    def DICELoss(self, X, Y, prob=None):
        disc_loss = torch.mean(torch.abs(F.softmax(X, dim=1) - F.softmax(Y, dim=1)))
        return disc_loss

