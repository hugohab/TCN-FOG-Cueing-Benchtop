import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FOGDataset(Dataset):
    def __init__(self, folder_path, normalize=True, debug_single_file=False):
        self.x_list = []
        self.y_list = []

        # Loop door alle .npz bestanden in de dataset map
        for file in os.listdir(folder_path):
            if file.endswith(".npz"):
                full_path = os.path.join(folder_path, file)
                data = np.load(full_path)

                x = data["xTensor"].astype(np.float32)  # (N,120,6)
                y = data["yTensor"].astype(np.int64)    # (N,)

                # Transpose naar (N, C, T)
                x = np.transpose(x, (0, 2, 1))

                self.x_list.append(x)
                self.y_list.append(y)

                # *** STOP after first file if debug enabled ***
                if debug_single_file:
                    print("Debug mode: loaded only 1 file.\n")
                    break
                

        # Combineer alle bestanden in één dataset
        self.x = np.concatenate(self.x_list, axis=0)
        self.y = np.concatenate(self.y_list, axis=0)

        # Normalisatie
        if normalize:
            mu = self.x.mean(axis=(0, 2), keepdims=True)
            sigma = self.x.std(axis=(0, 2), keepdims=True) + 1e-8
            self.x = (self.x - mu) / sigma

        # Converteer naar torch tensors
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
################################
# for only 1 npz
##############################
# import numpy as np # loads .npz files
# import torch # convert data to tensors
# from torch.utils.data import Dataset # base class for PyTorch datasets

# class FOGDataset(Dataset):
#     def __init__(self, npz_path, normalize=True):
#         data = np.load(npz_path)
#         self.x = data["xTensor"].astype(np.float32)  # (N,120,6) = (samples, time_steps, channels)
#         self.y = data["yTensor"].astype(np.int64)    # (N,) = labels

#         # (N, channels=6, timesteps=120)
#         # TCN expects input shaped like: (batch_size, channels, time_steps) 
#         # but the file provides: (batch_size, time_steps, channels)
#         self.x = np.transpose(self.x, (0, 2, 1)) 

#         if normalize:# z-score normalization per channel
#             mu = self.x.mean(axis=(0, 2), keepdims=True)
#             sigma = self.x.std(axis=(0, 2), keepdims=True) + 1e-8
#             self.x = (self.x - mu) / sigma

#         # Convert to PyTorch tensors, Required for using data in a PyTorch model.
#         self.x = torch.tensor(self.x)
#         self.y = torch.tensor(self.y)

#     def __len__(self): # tells DataLoader how many samples there are
#         return len(self.y)

#     def __getitem__(self, idx): # returns 1 sample (x, y) each time
#         return self.x[idx], self.y[idx]
