import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Noise + aliasing demo
# High-frequency sinusoid + noise (noise power 10 dB above signal)
# ------------------------------------------------------------

def alias_frequency(k, D):
    k_prime = (k * D) % 1.0
    k_alias = k_prime if k_prime <= 0.5 else 1.0 - k_prime
    return k_prime, k_alias


# Parameters
N = 4096
D = 4
k_high = 0.47  # cycles/sample (close to Nyquist)
n = np.arange(N)

# Clean high-frequency signal
signal = np.sin(2 * np.pi * k_high * n)

# ------------------------------------------------------------
# Add noise: 10 dB stronger than signal
# ------------------------------------------------------------
# Signal power of a sine wave with amplitude 1 is 0.5
signal_power = np.mean(signal**2)

noise_power = signal_power * 10.0  # +10 dB
noise_std = np.sqrt(noise_power)

rng = np.random.default_rng(0)
noise = noise_std * rng.standard_normal(N)

noisy_signal = signal + noise

# ------------------------------------------------------------
# Downsample (naive)
# ------------------------------------------------------------
noisy_ds = noisy_signal[::D]
m = np.arange(len(noisy_ds))

# Aliased frequency
k_prime, k_alias = alias_frequency(k_high, D)
alias_sine = np.sin(2 * np.pi * k_alias * m)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

# Show short windows for clarity
win_full = 300
win_ds = 300

axs[0].plot(n[:win_full], signal[:win_full], lw=1.5)
axs[0].set_title("Original high-frequency sinusoid (noise-free)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True, alpha=0.3)

axs[1].plot(n[:win_full], noisy_signal[:win_full], lw=1.0)
axs[1].set_title("Original signal + noise (noise power = signal + 10 dB)")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True, alpha=0.3)

axs[2].plot(m[:win_ds], noisy_ds[:win_ds], lw=1.5, label="Downsampled noisy signal")
axs[2].plot(m[:win_ds], alias_sine[:win_ds], "--", lw=2.0,
            label=f"Underlying aliased sinusoid (k_alias={k_alias:.3f})")
axs[2].set_title("After naive downsampling → low-frequency sinusoid emerges from noise")
axs[2].set_xlabel("Downsampled index")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.show()

print("Aliasing + noise summary:")
print(f"  Original frequency k       = {k_high:.4f} cycles/sample")
print(f"  Downsample factor D        = {D}")
print(f"  Aliased frequency k_alias  = {k_alias:.4f} cycles/sample (downsampled rate)")
print(f"  Signal power               = {signal_power:.4f}")
print(f"  Noise power                = {noise_power:.4f}  (+10 dB)")
print("")
print("Key takeaway:")
print("  Even when broadband noise dominates the full-rate signal,")
print("  deterministic high-frequency structure still aliases into")
print("  a coherent low-frequency sinusoid after downsampling.")

