#!/usr/bin/env python3
"""
1D demonstrations of detector artifacts BEFORE and AFTER naive downsampling.

Each row:
- Left: original sampled signal
- Right: naive downsampled version (stride, no filtering)

PRNU + downsampling will show the clearest "screen-door"-like aliasing.
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
DOWNSAMPLE = 4
x = np.arange(N)

rng = np.random.default_rng(42)

# Underlying smooth "scene"
scene = (
    0.6 * np.sin(2 * np.pi * x / 96.0) +
    0.3 * np.sin(2 * np.pi * x / 37.0) +
    0.2 * (x / N)
)
scene = normalize(scene)

# ------------------------------------------------------------
# 1) Detector grid itself
# ------------------------------------------------------------
detector_grid = np.zeros(N)
detector_grid[:] = 1.0

detector_grid_ds = detector_grid[::DOWNSAMPLE]

# ------------------------------------------------------------
# 2) Fixed-pattern noise (additive)
# ------------------------------------------------------------
fpn = 0.12 * np.sin(2 * np.pi * x / 8.0)     # periodic FPN
fpn_signal = normalize(scene + fpn)

fpn_ds = fpn_signal[::DOWNSAMPLE]

# ------------------------------------------------------------
# 3) PRNU (multiplicative)  <-- screen-door analog
# ------------------------------------------------------------
prnu_gain = 1.0 + 0.08 * rng.standard_normal(N)
prnu_signal = normalize(scene * prnu_gain)

prnu_ds = prnu_signal[::DOWNSAMPLE]

# ------------------------------------------------------------
# 4) Dead pixels / column-like effects (1D analog)
# ------------------------------------------------------------
dead_signal = scene.copy()

# Dead pixels
dead_indices = rng.choice(N, size=6, replace=False)
dead_signal[dead_indices] = 0.0

# Repeating "column" defects
for k in range(0, N, 32):
    dead_signal[k:k+2] *= 0.3

dead_signal = normalize(dead_signal)

dead_ds = dead_signal[::DOWNSAMPLE]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col")

titles = [
    "1) Detector grid itself",
    "2) Fixed-pattern noise (additive)",
    "3) PRNU (multiplicative)  ← screen-door analog",
    "4) Dead pixels / column effects",
]

signals = [
    (detector_grid, detector_grid_ds),
    (fpn_signal, fpn_ds),
    (prnu_signal, prnu_ds),
    (dead_signal, dead_ds),
]

for i, ((orig, ds), title) in enumerate(zip(signals, titles)):
    axes[i, 0].plot(x, orig, lw=1.5)
    axes[i, 0].set_title(f"{title} — original")
    axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(x[::DOWNSAMPLE], ds, lw=1.5, color="tab:red")
    axes[i, 1].set_title(f"{title} — downsampled ×{DOWNSAMPLE}")
    axes[i, 1].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Detector index")
axes[-1, 1].set_xlabel("Detector index (downsampled)")

plt.tight_layout()
plt.show()
