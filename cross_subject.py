import os
import copy
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import time
import torch
import numpy as np
from config import get_parser

from utils.utils import setup_seed
from utils.evaluate import evaluate
from utils.save import save_subject_results, save_final_results

from data_utils.get_dataset import get_dataset
from data_utils.MultiSourceLoader import MultiSourceLoader


from trainers import MSCGDATrainer
from models import MSPSDA

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


def cross_subject(args):
    """
    Main function to run the training process.
    """
    setup_seed(args.seed)
    os.makedirs(args.tmp_saved_path, exist_ok=True)
    
    # Load the dataset first in LOSO to avoid repeated loading for each train
    dataset = get_dataset(args)

    data = dataset["data"]
    labels = dataset["labels"]
    groups = dataset["groups"]

    results = []  # 用来存每一折的结果
    
    all_subjects = np.unique(groups[:, 0])

    for subject in all_subjects:
        print(f"--->受试者 {subject}")
        # 每一个受试者都重新定义种子。
        setup_seed(args.seed)
        if args.ablation == "num_sources":
            source_subjects = all_subjects[all_subjects != subject]
            selected_subjects = source_subjects[:args.num_sources]
            # print(selected_subjects)
            train_mask = np.isin(groups[:, 0], selected_subjects)
        else:
            train_mask = groups[:, 0] != subject
        # print(len(train_mask))
        # print(all_subjects)
        print(f"训练样本数: {np.sum(train_mask)}, 测试样本数: {np.sum(groups[:,0]==subject)}")
        train_dataset = {
            "data": data[train_mask],
            "labels": labels[train_mask],
            "groups": groups[train_mask]
        }

        val_dataset = {
            "data": data[groups[:,0] == subject],
            "labels": labels[groups[:,0] == subject],
            "groups": groups[groups[:,0] == subject]
        }
        train_loader, val_loader = MultiSourceLoader(args, train_dataset, val_dataset)()
        trainer = get_model_utils(args)
        
        start_time = time.time()
        best_val_accs, best_f1_scores, np_log = trainer.train(train_loader, val_loader)
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
    cross_subject(args)
