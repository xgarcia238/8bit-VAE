import glob
import pickle
import numpy as np
import os

import torch
import checkpoint
from checkpoint import Checkpoint
from torch.utils.data import Dataset

class Compositions(Dataset):
    """
    ----------------------------------------------------------------------------
    A dataset for compositions (i.e. scores). Could also in theory be used for
    for expressive performance.
    ----------------------------------------------------------------------------
    """
    def __init__(self, reference, root_dir):
        """
        ------------------------------------------------------------------------
        Args:
            reference (string): path to pickle file holding names of song chunks.
            root_dir  (string): path to root directory, containing the data.
        ------------------------------------------------------------------------
        """
        with open(os.path.join(root_dir,reference), 'rb') as f:
            self.idx_to_name = pickle.load(f)
        self.root_dir  = root_dir

    def __len__(self):
        return len(self.idx_to_name)

    def __getitem__(self, idx):
        name   = self.idx_to_name[idx]
        data_name = name + ".npy"
        target_name = name + "_target.npy"
        #data  = torch.load(os.path.join(self.root_dir, data_name)).astype(float)
        #score = torch.load(os.path.join(self.root_dir, target_name)).astype(long)
        data  = torch.tensor(np.load(os.path.join(self.root_dir, data_name)),dtype=torch.float32)
        score = torch.from_numpy(np.load(os.path.join(self.root_dir, target_name)))

        return data, score.long()

class CompactCompositions(Dataset):
    """
    ----------------------------------------------------------------------------
    A dataset for composition using an analog of Magenta's compact notation.
    ----------------------------------------------------------------------------
    """

    def __init__(self, root_dir):
        """
        ------------------------------------------------------------------------
        Args:
            reference (string): path to pickle file holding names of song chunks.
            root_dir  (string): path to root directory, containing the data.
        ------------------------------------------------------------------------
        """
        self.data = torch.load(os.path.join(root_dir, "data"))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].long()
