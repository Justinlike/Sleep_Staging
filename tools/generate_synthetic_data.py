import os
import numpy as np


def make_epoch_signal(fs=100, epoch_secs=30, stage=None):
    """Create a synthetic 30s EEG-like epoch.

    stage (int): optional sleep stage 0-4 to vary spectral content.
    """
    t = np.arange(0, epoch_secs, 1.0 / fs)

    # Base components: delta (0.5-4Hz), theta (4-8Hz), alpha (8-12Hz), spindle-ish (12-15Hz)
    sig = np.zeros_like(t)

    # Random amplitudes
    amp_delta = np.random.uniform(5, 20)
    amp_theta = np.random.uniform(1, 6)
    amp_alpha = np.random.uniform(0.5, 4)
    amp_spindle = np.random.uniform(0, 2)

    # Slightly vary by stage (rough heuristic)
    if stage is not None:
        if stage == 0:  # Wake: more alpha
            amp_alpha *= 1.5
            amp_delta *= 0.5
        elif stage == 1:  # N1: increased theta
            amp_theta *= 1.3
        elif stage == 2:  # N2: spindles
            amp_spindle *= 2.5
        elif stage == 3:  # N3: strong delta
            amp_delta *= 2.5
            amp_alpha *= 0.3
        elif stage == 4:  # REM: theta + some alpha
            amp_theta *= 1.3
            amp_alpha *= 1.1

    sig += amp_delta * np.sin(2 * np.pi * np.random.uniform(0.5, 2.5) * t)
    sig += amp_theta * np.sin(2 * np.pi * np.random.uniform(4, 7) * t)
    sig += amp_alpha * np.sin(2 * np.pi * np.random.uniform(8, 12) * t)
    sig += amp_spindle * np.sin(2 * np.pi * np.random.uniform(12, 15) * t)

    # Add low-frequency trend and random transient events
    sig += 0.2 * np.sin(2 * np.pi * 0.1 * t)  # slow baseline

    # Random gaussian noise
    noise = np.random.normal(0, 3.0, size=t.shape)
    sig = sig + noise

    # Small random artifacts
    if np.random.rand() < 0.02:
        # transient spike
        idx = np.random.randint(0, len(sig))
        width = int(fs * np.random.uniform(0.05, 0.5))
        sig[idx:idx+width] += np.random.uniform(50, 150)

    return sig.astype(np.float32)


def generate_and_save(output_dir, n_files=5, epochs_per_file=300, fs=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # approximate realistic class distribution (Wake, N1, N2, N3, REM)
    probs = np.array([0.15, 0.05, 0.55, 0.15, 0.10])
    probs = probs / probs.sum()

    samples_per_epoch = int(30 * fs)

    for f in range(1, n_files + 1):
        x = np.zeros((epochs_per_file, samples_per_epoch), dtype=np.float32)
        y = np.zeros((epochs_per_file,), dtype=np.int32)

        for e in range(epochs_per_file):
            label = np.random.choice(5, p=probs)
            y[e] = int(label)
            x[e] = make_epoch_signal(fs=fs, epoch_secs=30, stage=label)

        filename = os.path.join(output_dir, f"synthetic_{f}.npz")
        np.savez(filename, x=x, y=y, fs=fs)
        print(f"Saved {filename}  (x.shape={x.shape}, y.shape={y.shape})")


if __name__ == '__main__':
    # Generate ~100 sequences when seq_length=15 -> need 1500 epochs total
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_eeg')
    out_dir = os.path.abspath(out_dir)
    print('Output directory:', out_dir)
    generate_and_save(out_dir, n_files=5, epochs_per_file=300, fs=100, seed=42)
    print('Done.')
