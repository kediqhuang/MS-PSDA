import os
import copy
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import time
import torch
import numpy as np
from config import get_parser
from sklearn.preprocessing import MinMaxScaler
from utils.utils import setup_seed
from utils.evaluate import evaluate
from utils.save import save_subject_results, save_final_results

from data_utils.MultiSourceLoader import MultiSourceLoader
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset

from trainers import MSCGDATrainer
from models import MSPSDA

def get_data(args):
    if args.dataset_name.lower() == "seed3":
        num_classes = 3
        num_of_channels = 62
        SEED = SEEDFeatureDataset(args.seed3_path, sessions=[1, 2, 3])
        dataset = SEED.get_dataset()
        feature_dim = num_of_channels * SEED.get_feature_dim()
        data, int_labels, Group = dataset["data"], dataset["labels"], dataset["groups"]
        int_labels += 1

        data = data.reshape(-1, feature_dim)

        # 所有session合并 归一化
        subject_ids = Group[:, 0]  # subject ID

        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        for subj in np.unique(subject_ids):
            mask = subject_ids == subj
            data[mask] = min_max_scaler.fit_transform(data[mask])

        samples_per_session = data.shape[0] // 3
        data = data.reshape(3, samples_per_session, feature_dim)  # (3, n_per_session, 310)
        int_labels = int_labels.reshape(3, samples_per_session)
        Group = Group.reshape(3, samples_per_session, 3)

        src_indices = [si for si in range(3) if si + 1 != args.session]
        src_data = np.concatenate([data[s_i] for s_i in src_indices], axis=0)
        src_labels = np.concatenate([int_labels[s_i] for s_i in src_indices], axis=0)
        src_groups = np.concatenate([Group[s_i] for s_i in src_indices], axis=0)
        tgt_data = data[args.session - 1]
        tgt_labels = int_labels[args.session - 1]
        tgt_groups = Group[args.session - 1]

        src_one_hot_mat = np.eye(len(src_labels), num_classes)[src_labels].astype("float32")
        tgt_one_hot_mat = np.eye(len(tgt_labels), num_classes)[tgt_labels].astype("float32")
    elif args.dataset_name.lower() == "seed4":
        num_classes = 4
        num_of_channels = 62
        SEEDIV = SEEDIVFeatureDataset(args.seed4_path, sessions=[1, 2, 3])
        dataset = SEEDIV.get_dataset()
        feature_dim = num_of_channels * SEEDIV.get_feature_dim()
        data, int_labels, Group = dataset["data"], dataset["labels"], dataset["groups"]

        data = data.reshape(-1, feature_dim)

        # 所有session合并 归一化
        subject_ids = Group[:, 0]  # subject ID

        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        for subj in np.unique(subject_ids):
            mask = subject_ids == subj
            data[mask] = min_max_scaler.fit_transform(data[mask])

        samples_per_session = data.shape[0] // 3
        data = data.reshape(3, samples_per_session, feature_dim)  # (3, n_per_session, 310)
        int_labels = int_labels.reshape(3, samples_per_session)
        Group = Group.reshape(3, samples_per_session, 3)

        src_indices = [si for si in range(3) if si + 1 != args.session]
        src_data = np.concatenate([data[s_i] for s_i in src_indices], axis=0)
        src_labels = np.concatenate([int_labels[s_i] for s_i in src_indices], axis=0)
        src_groups = np.concatenate([Group[s_i] for s_i in src_indices], axis=0)
        tgt_data = data[args.session - 1]
        tgt_labels = int_labels[args.session - 1]
        tgt_groups = Group[args.session - 1]

        src_one_hot_mat = np.eye(len(src_labels), num_classes)[src_labels].astype("float32")
        tgt_one_hot_mat = np.eye(len(tgt_labels), num_classes)[tgt_labels].astype("float32")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_classes", num_classes)

    return {
        "data": src_data,
        "labels": src_one_hot_mat,
        "groups": src_groups,
    },{
        "data": tgt_data,
        "labels": tgt_one_hot_mat,
        "groups": tgt_groups,
    }

def get_model_utils(args):
    """
    Get model, optimizer, scheduler, and trainer.
    """
    params = {
        "input_dim": args.feature_dim, "num_classes": args.num_classes,
        "transfer_loss_type": args.transfer_loss_type, "max_iter": args.max_iter,
        "last_iter": args.last_iter, "num_src_clusters": args.num_src_clusters,
        "num_tgt_clusters": args.num_tgt_clusters, "num_sources": args.num_sources,
        "src_momentum": args.src_momentum, "tgt_momentum": args.tgt_momentum,
    }
    model = MSPSDA(**params).cuda()
    optimizer1 = torch.optim.Adam(model.get_step1_parameters(), lr=1e-3)
    optimizer2 = torch.optim.Adam(model.get_step2_parameters(), lr=1e-3)
    return MSCGDATrainer(
        model, optimizer1, optimizer2, lr_scheduler=None, max_iter=args.max_iter,
        last_iter=args.last_iter, transfer_loss_weight=args.transfer_loss_weight,
        constraint_loss_weight=args.constraint_loss_weight, dice_loss_weight=args.dice_loss_weight,
        cross_src_clu_loss_weight=args.cross_src_clu_loss_weight, early_stop=args.early_stop,
        log_interval=args.log_interval, branch=args.branch, mode=args.mode, sv_mode=args.sv_mode)


def cross_session(args):
    setup_seed(args.seed)
    os.makedirs(args.tmp_saved_path, exist_ok=True)

    # 加载按会话组织的数据列表
    src_dataset, tgt_dataset = get_data(args)  # 长度为3的列表

    train_data = src_dataset["data"]
    train_labels = src_dataset["labels"]
    train_groups = src_dataset["groups"]

    tgt_data = tgt_dataset["data"]
    tgt_labels = tgt_dataset["labels"]
    tgt_groups = tgt_dataset["groups"]

    results = []  # 用来存每一折的结果

    # all_subjects = np.unique(tgt_groups[:, 0])

    for subject in range(1, args.num_subjects + 1):
        print(f"--->受试者 {subject}")
        # 每一个受试者都重新定义种子。
        setup_seed(args.seed)

        print(f"训练样本数: {train_data.shape[0]}, 测试样本数: {np.sum(tgt_groups[:, 0] == subject)}")
        train_dataset = {
            "data": train_data,
            "labels": train_labels,
            "groups": train_groups
        }

        val_dataset = {
            "data": tgt_data[tgt_groups[:, 0] == subject],
            "labels": tgt_labels[tgt_groups[:, 0] == subject],
            "groups": tgt_groups[tgt_groups[:, 0] == subject]
        }
        train_loader, val_loader = MultiSourceLoader(args, train_dataset, val_dataset)()
        trainer = get_model_utils(args)

        start_time = time.time()
        sub_i = None
        best_val_accs, best_f1_scores, np_log = trainer.train(train_loader, val_loader, sub_i)
        train_time = time.time() - start_time

        # ================== Testing ==================
        # 加载最佳模型
        trainer.model.load_state(trainer.get_best_model_state())
        # print(trainer.model.classifier.P[:100])
        val_start = time.time()
        _, _ = evaluate(trainer.model, val_loader, args.branch, args.mode, args.sv_mode, args.device)
        val_time = (time.time() - val_start) * 1000

        entry = {
            "subject": subject,
            "train_time(s)": train_time,
        }
        # 动态添加每个最佳准确率
        for i, acc in enumerate(best_val_accs):
            entry[f"best_acc{i}"] = acc
            entry[f"best_f1_score{i}"] = best_f1_scores[i]
        results.append(entry)

        save_subject_results(args, subject, trainer, np_log)
        print(f"\n受试者 {subject} 完成:")
        for i, acc in enumerate(best_val_accs):
            print(f"  - 最佳验证准确率-{i}: {acc:.4f}")
        print(f"  - 训练时间: {train_time:.2f}s")
        print(f"  - 验证时间: {val_time:.2f}ms")

    save_final_results(args, results)
    print(f"\n{'=' * 60}")
    print("所有受试者训练完成！")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cross_session(args)
