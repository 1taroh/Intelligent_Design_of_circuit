import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import RCDataset

def plot_freq_response(rc_dataset:RCDataset, num_samples:int | list):
    fs = rc_dataset.fs.numpy()

    if isinstance(num_samples, int):
        indices = np.random.choice(len(rc_dataset), num_samples, replace=False)
    else:
        indices = num_samples

    H_samples = rc_dataset.H[indices].numpy()
    R_samples = rc_dataset.R[indices].numpy().flatten()
    C_samples = rc_dataset.C[indices].numpy().flatten()
    
    f_cut_min = 1 / (2 * np.pi * rc_dataset.R_range[1] * rc_dataset.C_range[1])
    f_cut_max = 1 / (2 * np.pi * rc_dataset.R_range[0] * rc_dataset.C_range[0])

    plt.figure(figsize=(10, 8))

    # Amplitude Plot
    plt.subplot(2, 1, 1)
    for Hs, R, C in zip(H_samples, R_samples, C_samples):
        plt.semilogx(fs, 20 * np.log10(np.abs(Hs)), label=f'R={R/1e3:.1f}k, C={C/1e-9:.1f}n')

    # Add boundary lines
    plt.axvline(f_cut_min, color='red', linestyle='--', alpha=0.3, label=f'f_cut_min ({f_cut_min:.1f} Hz)')
    plt.axvline(f_cut_max, color='blue', linestyle='--', alpha=0.3, label=f'f_cut_max ({f_cut_max:.1f} Hz)')

    plt.title('Frequency Response - Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xlim(fs.min(), fs.max())
    plt.ylim(-50, 10)
    plt.grid(True, which="both", ls="-")
    plt.legend()

    # Phase Plot
    plt.subplot(2, 1, 2)
    for Hs in H_samples:
        plt.semilogx(fs, np.angle(Hs))

    plt.axvline(f_cut_min, color='red', linestyle='--', alpha=0.3)
    plt.axvline(f_cut_max, color='blue', linestyle='--', alpha=0.3)

    plt.title('Frequency Response - Phase')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [deg]')
    plt.xlim(fs.min(), fs.max())
    plt.ylim(-np.pi/2-0.1, 0.1)
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()


def plot_freq_response_comparison(fs:torch.Tensor, Hs_actual:torch.Tensor, Hs_pred:torch.Tensor, Rs:torch.Tensor, Cs:torch.Tensor):
    """
    >>> (Rs, Cs), Hs_actual = test_dataset[idx]
    >>> Hs_pred = model([Rs, Cs])
    >>> plot_freq_response_comparison(fs, Hs_actual, Hs_pred, Rs, Cs)
    """
    fs = fs.detach().cpu().numpy()
    Hs_actual = Hs_actual.detach().cpu().numpy()
    Hs_pred = Hs_pred.detach().cpu().numpy()
    Rs = Rs.detach().cpu().numpy()
    Cs = Cs.detach().cpu().numpy()

    fig, ax = plt.subplots(2, len(Hs_actual), figsize = (5*len(Hs_actual), 10))
    for i, (H_actual, H_pred, R, C) in enumerate(zip(Hs_actual, Hs_pred, Rs, Cs)):
        ax[0, i].semilogx(fs, 20 * np.log10(np.abs(H_actual)), label='Actual')
        ax[1, i].semilogx(fs, np.angle(H_actual), label = 'Actual')
        ax[0, i].semilogx(fs, 20 * np.log10(np.abs(H_pred)), label='Predicted', linestyle="--")
        ax[1, i].semilogx(fs, np.angle(H_pred), label = 'Predicted', linestyle="--")

        ax[0, i].set_title(f'Actual (R={R/1e3:.1f} kΩ, C={C/1e-9:.1f} nF)')
        ax[0, i].legend()

        ax[0, i].grid(True, which="both", ls="-")
        ax[0, i].set_xlim(fs.min(), fs.max())
        ax[0, i].set_ylim(-50, 10)
        ax[0, i].set_xlabel('Frequency (Hz)')
        ax[0, i].set_ylabel('Magnitude (dB)')

        ax[1, i].grid(True, which="both", ls="-")
        ax[1, i].set_xlim(fs.min(), fs.max())
        ax[1, i].set_ylim(-np.pi/2 - 0.1, 0.1)
        ax[1, i].set_xlabel('Frequency (Hz)')
        ax[1, i].set_ylabel('Phase (rad)')

    plt.tight_layout()


if __name__ == "__main__":
    fs = torch.logspace(1, 5, 100)
    dataset = RCDataset(num_samples=1000, R_range=[10e3, 100e3], C_range=[1e-9, 10e-9], fs=fs)
    plot_freq_response(dataset, num_samples=5)


    # dataset = RCDataset(num_samples=1000, R_range=[10e3, 100e3], C_range=[1e-9, 10e-9], fs=fs)
    idx = [0, 1, 2]
    Hs_actual, circuit_parameters = dataset[idx]
    Rs = circuit_parameters[:, 0]      # Resistance values
    Cs = circuit_parameters[:, 1]      # Capacitance values
    plot_freq_response_comparison(dataset.fs, Hs_actual, Hs_actual, Rs, Cs)

    plt.show()
