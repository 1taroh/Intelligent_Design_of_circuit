import numpy as np
import torch
from torch.utils.data import Dataset

class RCDataset(Dataset):
    def __init__(self, num_samples, R_range, C_range, fs:torch.Tensor):
        """
        Args:
            num_samples: Number of data points to generate
            R_range: [min_R, max_R]
            C_range: [min_C, max_C]
            fs: The frequency array (Hz) to evaluate H for

            For fixed R, use R_range = [R, R].
        """
        self.num_samples = num_samples
        self.fs = fs
        self.ws = 2 * torch.pi * self.fs
        self.R_range = R_range
        self.C_range = C_range

        # R の生成
        if R_range[0] == R_range[1]:
            self.R = torch.full((num_samples, 1), R_range[0]) # for fixed R
        else:
            self.R = torch.distributions.Uniform(R_range[0], R_range[1]).sample((num_samples, 1))

        # C の生成
        if C_range[0] == C_range[1]:
            self.C = torch.full((num_samples, 1), C_range[0]) # for fixed C
        else:
            self.C = torch.distributions.Uniform(C_range[0], C_range[1]).sample((num_samples, 1))

        # Calculate Complex Transfer Function: H(jw) = 1 / (1 + jwRC)
        # Shape: (num_samples, len(fs))
        jwRC = 1j * self.ws * self.R * self.C
        self.H = 1 / (1 + jwRC)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # To match the expected format for plotting/analysis:
        # Returns (H, R, C)
        H = self.H[idx]   # Complex frequency response
        r = self.R[idx]   # Resistance value
        c = self.C[idx]   # Capacitance value

        circuit_parameters = torch.cat([r, c], dim=1)

        return H, circuit_parameters

if __name__ == "__main__":
    fs = np.logspace(1, 5, 100)
    dataset = RCDataset(num_samples=1000, R_range=[10e3, 100e3], C_range=[1e-9, 10e-9], fs=fs)
    idx = [0, 1, 2]                            
    Hs_actual, circuit_parameters = dataset[idx]
    Rs = circuit_parameters[:, 0]      # Resistance values
    Cs = circuit_parameters[:, 1]      # Capacitance values
    
    for i in idx:
        print(f"H: {Hs_actual[i]}, R: {Rs[i]}, C: {Cs[i]}")

