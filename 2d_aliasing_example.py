#!/usr/bin/env python3
"""
2D aliasing demo:
High-frequency 2D sinusoid + noise (noise power +10 dB vs signal),
then naive downsampling produces visible low-frequency moiré/shading.
Also shows low-pass + downsample as the fix.

Run: python aliasing_2d_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def normalize01(img):
    img = img.astype(np.float32)
    img -= img.min()
    d = img.max() - img.min()
    return img / d if d > 1e-12 else np.zeros_like(img)


def alias_freq_1d(k, D):
    k_prime = (k * D) % 1.0
    k_alias = k_prime if k_prime <= 0.5 else 1.0 - k_prime
    return k_prime, k_alias

# https://www.youtube.com/watch?v=fTJjPGaPsq4
def main():
    H, W = 1024, 1024
    D = 8

    # High spatial frequencies (cycles/pixel), close to Nyquist (0.5)
    kx_high = 0.47
    ky_high = 0.44

    yy, xx = np.mgrid[0:H, 0:W]

    # Clean high-frequency 2D signal (sum of two oriented sinusoids)
    signal = (
        np.sin(2*np.pi*(kx_high*xx + ky_high*yy)) +
        0.6*np.sin(2*np.pi*(0.49*xx - 0.41*yy))
    ).astype(np.float32)

    # Add noise: noise power = signal power * 10  (+10 dB)
    signal_power = np.mean(signal**2)
    noise_power = signal_power * 10.0
    noise_std = np.sqrt(noise_power)

    rng = np.random.default_rng(0)
    noise = (noise_std * rng.standard_normal((H, W))).astype(np.float32)

    noisy = signal + noise

    # Naive downsample (stride decimation)
    noisy_ds = noisy[::D, ::D]

    # Low-pass + downsample (anti-alias fix)
    sigma = 0.5 * D
    noisy_filt = gaussian_filter(noisy, sigma=sigma, mode="reflect")
    noisy_filt_ds = noisy_filt[::D, ::D]

    # Predict alias freqs for one component (intuition only)
    _, kx_alias = alias_freq_1d(kx_high, D)
    _, ky_alias = alias_freq_1d(ky_high, D)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    axs[0, 0].imshow(normalize01(signal), cmap="gray", interpolation="nearest")
    axs[0, 0].set_title("Clean high-frequency 2D sinusoid (full-rate)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(normalize01(noisy), cmap="gray", interpolation="nearest")
    axs[0, 1].set_title("Full-rate + noise (noise power = signal + 10 dB)")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(normalize01(noisy_ds), cmap="gray", interpolation="nearest")
    axs[1, 0].set_title(f"Naive downsample ×{D} (stride) → moiré / low-freq shading")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(normalize01(noisy_filt_ds), cmap="gray", interpolation="nearest")
    axs[1, 1].set_title(f"Low-pass (σ={sigma:.2f}) + downsample ×{D} → reduced aliasing")
    axs[1, 1].axis("off")

    plt.show()

    print("Summary:")
    print(f"  Image size: {H}x{W}")
    print(f"  Downsample factor D: {D}")
    print(f"  High freqs: kx={kx_high:.3f}, ky={ky_high:.3f} cycles/pixel (full-rate)")
    print(f"  Example aliased freqs (downsampled rate): kx_alias={kx_alias:.3f}, ky_alias={ky_alias:.3f} cycles/pixel")
    print(f"  Signal power: {signal_power:.4f}")
    print(f"  Noise power : {noise_power:.4f} (+10 dB)")


if __name__ == "__main__":
    main()



# https://www.youtube.com/watch?v=fTJjPGaPsq4