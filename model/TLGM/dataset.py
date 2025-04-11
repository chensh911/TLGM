import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, save_folder, seed, mode):
        # 加载节点表征
        representation_path = save_folder + f'HetSANN_{seed}_{mode}_rep.npy'
        label_idx_path = save_folder + f'HetSANN_{seed}_{mode}_label_idx.npy'
        self.representations = np.load(representation_path)
        self.label_idx = np.load(label_idx_path)

    def __len__(self):
        return self.representations.shape[0]

    def __getitem__(self, idx):
        rep = torch.tensor(self.representations[idx], dtype=torch.float)
        label_idx = torch.tensor(self.label_idx[idx], dtype=torch.long)
        return rep, label_idx
