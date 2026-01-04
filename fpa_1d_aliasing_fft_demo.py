#!/usr/bin/env python3
"""
1D FPA artifact + aliasing demo with FFTs.

For each of 4 cases:
  - Original (sampled at full rate)
  - Naive downsample (stride decimation, NO anti-alias)
  - Low-pass + downsample (anti-alias fix)

And for each of those:
  - Plot signal (space domain)
  - Plot FFT magnitude (frequency domain)

Key concept:
  Downsampling without band-limiting folds (aliases) high-frequency energy
  back into lower frequencies (spectral folding). Deterministic sensor artifacts
  (FPN/PRNU/defects) create coherent high-frequency components that fold into
  visible structure (the 2D analog is “screen door” / moiré).

Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    d = x.max() - x.min()
    if d < 1e-12:
        return np.zeros_like(x)
    return x / d


def fft_mag(x: np.ndarray):
    """
    Returns (freq, mag_db) for FFT magnitude in dB, centered at 0 with fftshift.
    freq is in cycles/sample (normalized), range approx [-0.5, 0.5).
    """
    x = x.astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x))
    mag = np.abs(X)
    mag_db = 20.0 * np.log10(mag + 1e-12)

    n = x.size
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))  # cycles/sample
    return freq, mag_db


def lowpass_then_downsample(x: np.ndarray, factor: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian low-pass then stride downsample.
    sigma is in samples.
    """
    xf = gaussian_filter1d(x, sigma=sigma, mode="reflect")
    xd = xf[::factor]
    return xf, xd


def make_signals(N=512, seed=42):
    rng = np.random.default_rng(seed)
    n = np.arange(N)

    # Underlying scene (band-limited-ish but includes some higher frequency content)
    scene = (
        0.55 * np.sin(2 * np.pi * n / 140.0) +
        0.30 * np.sin(2 * np.pi * n / 53.0)  +
        0.18 * np.sin(2 * np.pi * n / 1.2)  +  # closer to Nyquist than the others
        0.15 * (n / (N - 1))
    )
    scene = normalize01(scene)

    # 1) Detector grid itself (1D lattice/comb analogue)
    # In 1D, the "grid" is just the sample positions; to make it visible we show an impulse train.
    grid = np.zeros(N, dtype=np.float32)
    grid[::1] = 1.0  # impulses at every detector/sample
    grid = normalize01(grid)

    # 2) Fixed-pattern noise (FPN): additive offsets (often periodic + random per-pixel bias)
    fpn = 0.16 * np.sin(2 * np.pi * n / 8.0) + 0.05 * rng.standard_normal(N)
    fpn_signal = normalize01(scene + fpn)

    # 3) PRNU: multiplicative pixel gain variation (screen-door analog)
    prnu_gain = 1.0 + 0.10 * rng.standard_normal(N)  # per-pixel gain
    prnu_signal = normalize01(scene * prnu_gain)

    # 4) Dead pixels + "column-like" effects (1D analog)
    dead = scene.copy()
    dead_idx = rng.choice(N, size=10, replace=False)
    dead[dead_idx] = 0.0  # dead/stuck-low pixels

    # repeating defect blocks (column-ish in 2D → repeating stripes; in 1D → repeating segments)
    for k in range(0, N, 64):
        dead[k:k+3] *= 0.25

    dead_signal = normalize01(dead)

    return n, scene, {
        "Detector grid itself": grid,
        "Fixed-pattern noise (additive)": fpn_signal,
        "PRNU (multiplicative)  ← screen-door analog": prnu_signal,
        "Dead pixels / column-like effects": dead_signal,
    }


def main():
    # ---- Tunables ----
    N = 512
    D = 4  # downsample factor
    # Gaussian sigma rule of thumb: about 0.5*D samples (adjust if you want sharper/blurrier)
    sigma = 0.5 * D

    n, scene, signals = make_signals(N=N, seed=42)

    # Build plots: 4 rows (cases) x 2 columns (space domain + FFT domain),
    # and within each subplot overlay 3 curves: original, naive downsample, lowpass+downsample.
    fig, axes = plt.subplots(len(signals), 2, figsize=(16, 12))
    if len(signals) == 1:
        axes = np.array([axes])

    for row, (name, x) in enumerate(signals.items()):
        # Space-domain versions
        x_orig = x
        x_ds = x_orig[::D]
        x_filt, x_filt_ds = lowpass_then_downsample(x_orig, factor=D, sigma=sigma)

        # -------- Space domain plot --------
        ax = axes[row, 0]
        ax.plot(n, x_orig, lw=1.4, label="Original (full-rate)")
        ax.plot(n[::D], x_ds, lw=1.4, label=f"Naive downsample ×{D} (stride)", alpha=0.9)
        ax.plot(n[::D], x_filt_ds, lw=1.6, label=f"Low-pass (σ={sigma:.2f}) + downsample ×{D}", alpha=0.9)

        ax.set_title(f"{name} — space domain")
        ax.set_ylabel("Amplitude (normalized)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, N - 1)

        # -------- Frequency domain plot --------
        axf = axes[row, 1]

        f0, m0 = fft_mag(x_orig)
        f1, m1 = fft_mag(x_ds)
        f2, m2 = fft_mag(x_filt_ds)

        axf.plot(f0, m0, lw=1.4, label="Original (full-rate)")
        axf.plot(f1, m1, lw=1.4, label=f"Naive downsample ×{D}", alpha=0.9)
        axf.plot(f2, m2, lw=1.6, label="Low-pass + downsample", alpha=0.9)

        axf.set_title(f"{name} — FFT magnitude (dB)")
        axf.set_xlabel("Frequency (cycles/sample)")
        axf.set_ylabel("Magnitude (dB)")
        axf.grid(True, alpha=0.25)

        # Mark Nyquist of full-rate and of downsampled rate (relative to full-rate axis)
        # Full-rate Nyquist is ±0.5 cycles/sample.
        # After downsample by D, new Nyquist is ±0.5 cycles/(downsampled sample) which corresponds
        # to ±(0.5 / D) in full-rate cycles/sample if you compare to original scale.
        axf.axvline(0.5 / D, linestyle="--", alpha=0.35)
        axf.axvline(-0.5 / D, linestyle="--", alpha=0.35)
        axf.text(0.5 / D, axf.get_ylim()[0] + 3, "new Nyquist", rotation=90, va="bottom", ha="left", alpha=0.6)

        # Keep frequency axis centered
        axf.set_xlim(0, 0.5)

    # One shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = "fpa_1d_aliasing_fft_demo.png"
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"Saved: {out}")
    print(f"Try changing D={D} and sigma={sigma} to see folding change.")


if __name__ == "__main__":
    main()
