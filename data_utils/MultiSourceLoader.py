import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.cluster import DBSCAN
from dataset import MultiSourceDataset, SMDataset

class MultiSourceLoader(object):
    def __init__(self, args, train_dataset=None, val_dataset=None):
        super(MultiSourceLoader, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.clustering_method = getattr(args, "clustering_method", "dbscan")
        if self.clustering_method not in ("dbscan", "trial"):
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        if self.clustering_method == "dbscan":
            self.best_cluster_params = {"eps": 1.0, "min_samples": 3}
            print(self.best_cluster_params)

    def __call__(self):
        train_loader = self.create_train_loader()
        val_loader = self.create_val_loader()
        return train_loader, val_loader
    
    def _perform_clustering(self, X, Y, trial_ids=None):
        if self.clustering_method == "trial":
            if trial_ids is None:
                raise ValueError("Trial clustering requires trial IDs in groups[:, 1]")
            unique_trials, labels = np.unique(trial_ids, return_inverse=True)
            return len(unique_trials), X, Y, labels.astype(np.int64)
        clustering = DBSCAN(**self.best_cluster_params).fit(X)
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # print(num_clusters)
        return num_clusters, X, Y, labels

    def create_train_loader(self):
        data = self.train_dataset["data"]
        labels = self.train_dataset["labels"]
        subjects = self.train_dataset["groups"][:, 0]

        cluster_results = {}
        min_clusters = float('inf')

        setattr(self.args, "num_sources", len(np.unique(subjects)))
        for subject_id in np.unique(subjects):
            mask = subjects == subject_id
            n, x, y, c = self._perform_clustering(
                data[mask], labels[mask], self.train_dataset["groups"][mask, 1])
            cluster_results[subject_id] = (n, x, y, c)
            min_clusters = min(min_clusters, n)

        setattr(self.args, "num_src_clusters", min_clusters)

        source_domains = []
        for subject_id, (n, x, y, c) in cluster_results.items():
            print(n)
            # Filters the data to ensure each subject has the same number of clusters.
            cluster_counts = Counter(c)
            top_clusters = [cluster for cluster, _ in cluster_counts.most_common(min_clusters)]
            mask = np.isin(c, top_clusters)

            source_features = torch.from_numpy(x[mask]).type(torch.Tensor)
            source_labels = torch.from_numpy(y[mask]) 
            
            # Reorders cluster IDs to ensure consistency.
            source_cluster = c[mask]
            unique_clusters = np.unique(source_cluster)
            cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}

            source_cluster = torch.tensor([cluster_mapping[label] for label in source_cluster], dtype=torch.long)

            source_domains.append({
                "x": source_features,
                "y": source_labels,
                "cluster": source_cluster
            })
        
        source_datasets = MultiSourceDataset(source_domains)
        source_loader = DataLoader(
            dataset=source_datasets,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        return source_loader
    
    def create_val_loader(self):
        data = self.val_dataset["data"]
        labels = self.val_dataset["labels"]
        
        n, x, y, c = self._perform_clustering(data, labels, self.val_dataset["groups"][:, 1])
        
        setattr(self.args, "num_tgt_clusters", n)
        val_dataset = SMDataset(
            torch.from_numpy(x).type(torch.Tensor),
            torch.from_numpy(y),
            torch.from_numpy(c)
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True
        )
        return val_loader
