#!/usr/bin/env python3
"""
1D demonstrations of common FPA artifacts that lead to
screen-door / moiré effects when resampled.

Each subplot shows a different phenomenon:
1) Detector grid itself
2) Fixed-pattern noise (FPN)
3) Pixel response non-uniformity (PRNU)
4) Dead pixels / column-like effects (1D analog)
"""

import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    x = x.astype(np.float32)
    x -= x.min()
    d = x.max() - x.min()
    if d < 1e-12:
        return np.zeros_like(x)
    return x / d


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
N = 256
x = np.arange(N)

rng = np.random.default_rng(42)

# A smooth underlying "scene" signal
scene = (
    0.6 * np.sin(2 * np.pi * x / 96.0) +
    0.3 * np.sin(2 * np.pi * x / 37.0) +
    0.2 * (x / N)
)
scene = normalize(scene)

# ------------------------------------------------------------
# 1) Detector grid itself
# ------------------------------------------------------------
# Pure sampling lattice (impulse train)
detector_grid = np.zeros(N)
detector_grid[::1] = 1.0  # every detector location
# (in 1D this looks trivial, but it's the root cause)

# ------------------------------------------------------------
# 2) Fixed-pattern noise (additive)
# ------------------------------------------------------------
# Additive offsets that repeat spatially
fpn = 0.08 * np.sin(2 * np.pi * x / 8.0)      # periodic component
fpn += 0.04 * rng.standard_normal(N)          # random per-pixel offset
fpn_signal = normalize(scene + fpn)

# ------------------------------------------------------------
# 3) PRNU (multiplicative)
# ------------------------------------------------------------
# Pixel-to-pixel gain variation
prnu_gain = 1.0 + 0.05 * rng.standard_normal(N)
prnu_signal = normalize(scene * prnu_gain)

# ------------------------------------------------------------
# 4) Dead pixels / column effects (1D analog)
# ------------------------------------------------------------
dead_signal = scene.copy()

# Dead pixels
dead_indices = rng.choice(N, size=6, replace=False)
dead_signal[dead_indices] = 0.0

# "Column" effects in 1D = repeating stuck offsets
for k in range(0, N, 32):
    dead_signal[k:k+2] *= 0.3

dead_signal = normalize(dead_signal)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axes[0].stem(x, detector_grid, basefmt=" ", linefmt="k-", markerfmt=" ")
axes[0].set_title("1) Detector grid itself (sampling lattice)")
axes[0].set_ylabel("Amplitude")

axes[1].plot(x, fpn_signal, color="tab:blue")
axes[1].set_title("2) Fixed-pattern noise (additive offsets)")
axes[1].set_ylabel("Amplitude")

axes[2].plot(x, prnu_signal, color="tab:green")
axes[2].set_title("3) PRNU (multiplicative pixel gain variation)")
axes[2].set_ylabel("Amplitude")

axes[3].plot(x, dead_signal, color="tab:red")
axes[3].set_title("4) Dead pixels / column-like effects (1D analog)")
axes[3].set_ylabel("Amplitude")
axes[3].set_xlabel("Detector index")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
