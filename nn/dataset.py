import torch
from torch.utils.data.dataset import Dataset

import pandas as pd


class DriverDataset(Dataset):
    """Dataset of the training data provided by the CI course"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file containing training data.
        """
        df = pd.read_csv(csv_file)
        # Drop speed column
        df1 = df.drop(['SPEED'], axis=1)
        # Drop last row
        df1.drop(df1.tail(1).index, inplace=True)
        self.data = df1.as_matrix()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx, :3]), torch.Tensor(self.data[idx, 3:])
