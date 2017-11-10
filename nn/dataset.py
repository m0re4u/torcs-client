import torch
from torch.utils.data.dataset import Dataset

import pandas as pd


class DriverDataset(Dataset):
    """Dataset of the training data provided by the CI course"""

    def __init__(self, csv_file, skip_column=None):
        """
        Args:
            csv_file (string): Path to the csv file containing training data.
        """
        df = pd.read_csv(csv_file)
        if skip_column is not None:
            # Drop speed column
            df.drop([skip_column], axis=1, inplace=True)
        # Drop last row
        df.drop(df.tail(1).index, inplace=True)
        self.data = df.as_matrix()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx, :3]), torch.Tensor(self.data[idx, 3:])
