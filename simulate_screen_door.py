#!/usr/bin/env python3
"""
Simulate "screen door" / moiré from naive downsampling of an FPA-like image,
then fix it with a low-pass (anti-alias) filter before decimation.

Outputs:
- Displays a 2x2 figure:
  (1) Original synthetic FPA image
  (2) Naively downsampled (aliasing / screen door)
  (3) Low-pass filtered before downsample
  (4) Filtered + downsampled (reduced aliasing)
- Also saves "screen_door_demo.png" in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    denom = (x.max() - x.min())
    if denom < 1e-12:
        return np.zeros_like(x)
    return x / denom


def generate_scene(n: int, seed: int = 0) -> np.ndarray:
    """
    Create a synthetic scene with:
    - Smooth background
    - A few edges (high contrast)
    - High spatial frequencies near Nyquist (to provoke aliasing)
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)

    # Smooth ramp + vignette-ish term
    ramp = 0.25 * (x / (n - 1)) + 0.15 * (y / (n - 1))
    cx, cy = (n - 1) / 2.0, (n - 1) / 2.0
    r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (cx**2 + cy**2)
    vignette = 0.15 * (1.0 - r2)

    # Add sharp edges: rectangle + diagonal edge
    rect = ((x > 0.18 * n) & (x < 0.82 * n) & (y > 0.35 * n) & (y < 0.65 * n)).astype(np.float32)
    diag = (y > (0.6 * x + 0.1 * n)).astype(np.float32)

    edges = 0.45 * rect - 0.25 * diag

    # High-frequency content near Nyquist: sinusoids + slight rotation to create moiré after decimation
    # Frequencies chosen to be "dangerous" when downsampling by an integer factor.
    hf1 = 0.12 * np.sin(2.0 * np.pi * (0.47 * x + 0.13 * y))   # close to Nyquist in x
    hf2 = 0.10 * np.sin(2.0 * np.pi * (0.09 * x + 0.49 * y))   # close to Nyquist in y
    hf3 = 0.08 * np.sin(2.0 * np.pi * (0.33 * x + 0.33 * y))   # diagonal-ish

    # Small noise (like sensor noise)
    noise = 0.02 * rng.standard_normal((n, n)).astype(np.float32)

    scene = ramp + vignette + edges + hf1 + hf2 + hf3 + noise
    return normalize01(scene)


def apply_fpa_artifacts(img: np.ndarray, seed: int = 1) -> np.ndarray:
    """
    Add FPA-like artifacts:
    - PRNU (pixel response non-uniformity) as a multiplicative pixel-wise gain
    - Row/column pattern noise
    - A subtle pixel-grid visibility term (simulates fill-factor / pixel boundary emphasis)
    """
    rng = np.random.default_rng(seed)
    n = img.shape[0]

    # PRNU (multiplicative gain)
    prnu = 1.0 + 0.03 * rng.standard_normal((n, n)).astype(np.float32)

    # Row/column fixed pattern components
    row = 0.015 * rng.standard_normal((n, 1)).astype(np.float32)
    col = 0.015 * rng.standard_normal((1, n)).astype(np.float32)

    # Subtle "pixel boundary" emphasis: a weak checkerboard-like lattice
    yy, xx = np.mgrid[0:n, 0:n]
    lattice = ((xx + yy) % 2).astype(np.float32)  # 0/1 checkerboard
    lattice = (lattice - 0.5) * 0.03              # centered, weak amplitude

    fpa = img * prnu + row + col + lattice
    return normalize01(fpa)


def downsample_stride(img: np.ndarray, factor: int) -> np.ndarray:
    """Naive decimation by striding (guaranteed aliasing if not band-limited)."""
    return img[::factor, ::factor]


def lowpass_then_downsample(img: np.ndarray, factor: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Low-pass filter then downsample.
    Returns (filtered, downsampled).
    """
    filtered = gaussian_filter(img, sigma=sigma, mode="reflect")
    ds = downsample_stride(filtered, factor)
    return filtered, ds


def main():
    # Parameters
    n = 1024
    factor = 4

    # Anti-alias sigma: rule-of-thumb ~0.5*factor pixels (tune if desired)
    sigma = 0.5* factor

    scene = generate_scene(n=n, seed=0)
    fpa = apply_fpa_artifacts(scene, seed=1)

    naive_ds = downsample_stride(scene, factor=factor)
    filtered, filtered_ds = lowpass_then_downsample(fpa, factor=factor, sigma=sigma)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes.ravel()

    ax[0].imshow(fpa, cmap="gray", interpolation="nearest")
    ax[0].set_title("Original synthetic FPA image (scene + PRNU/FPN + weak lattice)")
    ax[0].axis("off")

    ax[1].imshow(naive_ds, cmap="gray", interpolation="nearest")
    ax[1].set_title(f"Naive downsample by {factor}x (stride) → aliasing / screen door")
    ax[1].axis("off")

    ax[2].imshow(filtered, cmap="gray", interpolation="nearest")
    ax[2].set_title(f"Low-pass filtered (Gaussian σ={sigma:.2f}) before downsample")
    ax[2].axis("off")

    ax[3].imshow(filtered_ds, cmap="gray", interpolation="nearest")
    ax[3].set_title(f"Low-pass + downsample {factor}x → reduced aliasing")
    ax[3].axis("off")

    plt.tight_layout()
    out = "screen_door_demo.png"
    plt.savefig(out, dpi=200)
    plt.show()

    print(f"\nSaved figure to: {out}")
    print(f"Tip: try changing factor={factor} and sigma={sigma} to see the effect.\n")


if __name__ == "__main__":
    main()
