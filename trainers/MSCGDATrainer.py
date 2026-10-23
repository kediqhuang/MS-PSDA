import torch
import numpy as np
import copy
import math
from sklearn.metrics import accuracy_score, f1_score

class MSCGDATrainer(object):
    def __init__(self, 
                 model, 
                 optimizer1,
                 optimizer2,
                 lr_scheduler = None,
                 max_iter: int = 1000,
                 last_iter: int = 0,
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 transfer_loss_weight: int = 1,
                 constraint_loss_weight: int = 0.01,
                 dice_loss_weight: int = 0.1,
                 cross_src_clu_loss_weight: int = 0.1,
                 branch: str = "",
                 mode: str = "",
                 sv_mode: str = "",
                 device: str = "cuda:0",
                 **kwargs):
        super().__init__()

        self.model = model
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.lr_scheduler = lr_scheduler

        self.log_interval = log_interval
        self.early_stop = early_stop
        self.max_iter = max_iter
        self.last_iter = last_iter

        self.transfer_loss_weight = transfer_loss_weight
        self.constraint_loss_weight = constraint_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.cross_src_clu_loss_weight = cross_src_clu_loss_weight

        self.branch = branch
        self.mode = mode
        self.sv_mode = sv_mode

        self.device = device
        self.stop = 0
        self.best_model_state = None

        self.src_num_clusters_list = []

    def get_model_state(self):
        if hasattr(self.model, "get_state"):
            return self.model.get_state()
        return self.model.state_dict()
    
    def get_best_model_state(self):
        return self.best_model_state

    def train(self, source_loaders, target_loader, sub_i=None):

        best_accs = None
        best_f1s = None
        log = []
        primary_idx = 1

        self.pre_processing(source_loaders)

        source_iter = iter(source_loaders)
        target_iter = iter(target_loader)
        src_i = 0
        if sub_i is not None:
            src_i = sub_i
        for it in range(self.max_iter):

            self.model.train()
            try:
                src_data, src_label, src_cluster = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loaders)
                src_data, src_label, src_cluster = next(source_iter)
            try:
                tgt_data, _, tgt_cluster = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _, tgt_cluster = next(target_iter)

            offset = 0
            cross_src_cluster_list = []
            for snc_idx, snc in enumerate(self.src_num_clusters_list):
                cross_src_cluster_list.append(src_cluster[:, snc_idx] + offset)
                offset += snc
            cross_src_cluster = torch.cat(cross_src_cluster_list, dim=0)

            src_data, src_label = src_data.to(
                self.device), src_label.to(self.device)
            tgt_data = tgt_data.to(self.device)
            src_cluster = src_cluster.to(self.device)
            tgt_cluster = tgt_cluster.to(self.device)
            cross_src_cluster = cross_src_cluster.to(self.device)

            a = 2 / (1 + math.exp(-10 * it / (self.max_iter - self.last_iter))) - 1
            gamma = a * self.dice_loss_weight

            # Phase 1: classifier
            cls_loss, cross_src_clu_loss, transfer_loss, cons_loss  = (
                self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster, cross_src_cluster, src_i, step="classifier"))
            loss = cls_loss + self.cross_src_clu_loss_weight * cross_src_clu_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss
            self.optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer1.step()

            # Phase 2: cluster
            src_clu_loss, tgt_clu_loss, dice_loss = (
                self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster, cross_src_cluster, src_i, step="cluster"))
            self.optimizer2.zero_grad()
            loss2 = src_clu_loss + tgt_clu_loss + gamma * dice_loss
            loss2.backward()
            self.optimizer2.step()

            target_accs, target_f1s = self.test(target_loader)
            if best_accs is None:
                best_accs = [-1.0] * len(target_accs)
                best_f1s = [-1.0] * len(target_accs)
            improved, best_accs, best_f1s = self._update_best_model(target_accs, target_f1s, best_accs, best_f1s, primary_idx=primary_idx)

            iter_losses = {
                'cls_l': cls_loss.detach().item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
                'CDC_l': cross_src_clu_loss.detach().item() if isinstance(cross_src_clu_loss, torch.Tensor) else cross_src_clu_loss,
                'tran_l': transfer_loss.detach().item() if isinstance(transfer_loss, torch.Tensor) else transfer_loss,
                'cons_l': cons_loss.detach().item() if isinstance(cons_loss, torch.Tensor) else cons_loss,
                "IDCs_l": src_clu_loss.detach().item() if isinstance(src_clu_loss, torch.Tensor) else src_clu_loss,
                "IDCt_l": tgt_clu_loss.detach().item() if isinstance(tgt_clu_loss, torch.Tensor) else tgt_clu_loss,
                "dice_l": dice_loss.detach().item(),
                # "gamma": gamma,
            }

            log_entry = [v for _, v in iter_losses.items()] + best_accs + best_f1s + target_accs + target_f1s
            log.append(log_entry)

            info = self._log_training_info(it, iter_losses, target_accs, best_accs)
            if sub_i is None:
                src_i = (src_i + 1) % source_loaders.dataset.num_domains
            if self._should_early_stop(best_accs, primary_idx=primary_idx):
                print(info)
                break
        return best_accs, best_f1s, np.array(log, dtype=float)
    
    @torch.no_grad()
    def test(self, dataloader):
        self.model.eval()
        feature, labels, _ = dataloader.dataset.get_data()
        labels = np.argmax(labels.numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device), branch=self.branch, mode=self.mode, sv_mode=self.sv_mode)
        outputs = []
        f1_outputs = []
        for i, y_pred in enumerate(y_preds):
            outputs.append(accuracy_score(labels, y_pred.cpu().numpy()) * 100.)
            f1_outputs.append(f1_score(labels, y_pred.cpu().numpy(), average='macro') * 100.)
        return outputs, f1_outputs

    def pre_processing(self, source_loaders):
        dataset = source_loaders.dataset 

        src_datas = []
        src_clusters = []
        src_labels = []
        for domain_idx in range(dataset.num_domains):
            x, y, cluster = dataset.get_single_domain(domain_idx)
            src_datas.append(x.to(self.device))
            src_clusters.append(cluster)
            src_labels.append(y)
        offset = 0
        cross_src_clusters = []
        for cluster in src_clusters:
            cross_src_clusters.append(cluster+offset)
            offset += len(np.unique(cluster))
            self.src_num_clusters_list.append(len(np.unique(cluster)))
        cross_src_clusters_merged = torch.cat(cross_src_clusters, dim=0)
        self.model.initialize_cluster(src_datas, src_clusters, src_labels, self.src_num_clusters_list, cross_src_clusters_merged)

    def _update_best_model(self, current_accs, current_f1s, best_accs, best_f1s, primary_idx=0):
        """更新最佳模型状态"""
        if isinstance(current_accs, (float, int)):
            current_accs = [current_accs]
            best_accs = [best_accs] if isinstance(best_accs, (float, int)) else best_accs
        improved = False
        for i, acc in enumerate(current_accs):
            if acc > best_accs[i]:
                best_accs[i] = acc
                best_f1s[i] = current_f1s[i]
                if i == primary_idx:
                    improved = True

        if improved:
            self.best_model_state = copy.deepcopy(self.get_model_state())
            self.stop = 0
        else:
            self.stop += 1

        return improved, best_accs, best_f1s
    
    def _should_early_stop(self, best_accs, primary_idx=0):
        """判断是否应该早停"""
        if self.early_stop == 0:
            return False
        flag = True
        if 100 - best_accs[primary_idx] > 1e-3:
            flag = False
        # for acc in best_accs:
        #     if 100 - acc > 1e-3:
        #         flag = False
        return (self.stop >= self.early_stop) or flag

    def _log_training_info(self, iteration, loss_dict, target_accs, best_accs):
        """打印训练信息"""
        info = f'Iter: [{iteration + 1:2d}/{self.max_iter}]'
        for key, value in loss_dict.items():
            info += f', {key}: {value:.4f}'
        info += f', target_accs: {[f"{a:.2f}" for a in target_accs]}'
        info += f', best_accs: {[f"{b:.2f}" for b in best_accs]}'
        if (iteration + 1) % self.log_interval == 0 or iteration == 0:
            print(info)
        return info
