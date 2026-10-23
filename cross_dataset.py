import os
import copy
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import time
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import get_parser
from utils.utils import setup_seed
from utils.evaluate import evaluate
from utils.save import save_subject_results, save_final_results
from data_utils.MultiSourceLoader import MultiSourceLoader
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset

from trainers import MSCGDATrainer
from models import MSPSDA

def get_seed_data(args, dataset_name):
    if dataset_name.lower() == "seed3":
        dataset = SEEDFeatureDataset(args.seed3_path, sessions=[1, 2, 3]).get_dataset()
        data, int_labels, Group = dataset["data"], dataset["labels"], dataset["groups"]
        data = data.reshape(-1, 310)
        int_labels += 1
        sGroup = Group[:, 0]

    elif dataset_name.lower() == "seed4":
        dataset = SEEDIVFeatureDataset(args.seed4_path, sessions=[1, 2, 3]).get_dataset()
        data, int_labels, Group = dataset["data"], dataset["labels"], dataset["groups"]
        data = data.reshape(-1, 310)
        lookup = np.array([1, 0, 0, 2])  # 0→1, 1→0, 2→0, 3→2
        int_labels = lookup[int_labels]
        sGroup = Group[:, 0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    for sid in np.unique(sGroup):
        data[sGroup == sid] = scaler.fit_transform(data[sGroup == sid])

    one_hot_labels = np.eye(args.num_classes)[int_labels].astype("float32")
    return data, one_hot_labels, Group

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


def cross_dataset(args, source_dataset="seed3", target_dataset="seed4"):
    """
    Train the model for cross-dataset adaptation.
    """
    # Set random seed
    setup_seed(args.seed)
    os.makedirs(args.tmp_saved_path, exist_ok=True)

    source_data, source_labels, source_groups = get_seed_data(args, source_dataset)
    target_data, target_labels, target_groups = get_seed_data(args, target_dataset)
    
    # 对受试者重新进行编号以混合3个session的数据
    source_groups[:, 0] = (source_groups[:, 2] - 1) * 15 + source_groups[:, 0]
    target_groups[:, 0] = (target_groups[:, 2] - 1) * 15 + target_groups[:, 0]

    results = []  # 用来存每一折的结果
    for subject in range(1, len(np.unique(target_groups[:, 0])) + 1):
        print(f"开始适应 {target_dataset} 受试者 {subject}")

        # 每一个受试者都重新定义种子。
        setup_seed(args.seed)
        val_mask = target_groups[:, 0] == subject
        train_dataset = {
            "data": source_data,
            "labels": source_labels,
            "groups": source_groups,
        }

        val_dataset = {
            "data": target_data[val_mask],
            "labels": target_labels[val_mask],
            "groups": target_groups[val_mask],
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
    print(f"\n{'='*60}")
    print("所有受试者训练完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    setattr(args, "num_classes", 3)
    setattr(args, "num_subjects", 45) # three sessions
    setattr(args, "feature_dim", 310)
    setattr(args, "num_sources", 45)
    base_logs = args.tmp_saved_path
    for source_dataset, target_dataset in [("seed3", "seed4"), ("seed4", "seed3")]:
        tmp_saved_path = os.path.join(base_logs,
                                      "cross_dataset",
                                      f"{source_dataset.upper()}_to_{target_dataset.upper()}",
                                      f"seed_{args.seed}")
        setattr(args, "tmp_saved_path", tmp_saved_path)
        
        cross_dataset(args, source_dataset=source_dataset, target_dataset=target_dataset)
