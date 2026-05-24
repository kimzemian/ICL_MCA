import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Normalizer:
    """Base class for normalization strategies."""

    def __init__(self):
        self.stats = {}

    def fit(self, train_ds):
        """Compute normalization statistics from training data."""
        raise NotImplementedError

    def transform(self, X):
        """Apply normalization to data."""
        raise NotImplementedError

    def fit_transform(self, train_ds):
        """Fit and transform in one step."""
        self.fit(train_ds)
        return self


class SecondNormalizer(Normalizer):
    def fit(self, train_X):
        """
        Compute mean and std for each feature independently.
        Args:
            train_X: numpy array of shape (n_samples, seq_len, num_levels, 7) - raw data
        """
        # Apply log transformations to training data
        X_transformed = self._log_transform(train_X)

        # Reshape to compute global statistics
        X_flat = X_transformed.reshape(
            -1, X_transformed.shape[-1]
        )  # (n_samples * seq_len * num_levels, 7)
        self.mean_ = np.mean(X_flat, axis=0)  # (7,)
        self.std_ = np.std(X_flat, axis=0) + 1e-8  # (7,)
        return self

    def _log_transform(self, X):
        """
        Helper to apply log transformations only.
        Handles both shapes:
        - (seq_len, num_levels, 7) for single sample transform
        - (n_samples, seq_len, num_levels, 7) for batch fit
        """
        X = X.copy()

        # Determine if we have 0batch dimension
        is_batch = X.ndim == 4

        if is_batch:
            # Extract best bid and best ask from level 0
            best_bid = X[:, :, 0, 0]  # (n_samples, seq_len)
            best_ask = X[:, :, 0, 3]  # (n_samples, seq_len)
            p_mid = (best_bid + best_ask) / 2.0
            p_mid_expanded = p_mid[:, :, np.newaxis]  # (n_samples, seq_len, 1)
        else:
            # Extract best bid and best ask from level 0
            best_bid = X[:, 0, 0]  # (seq_len,)
            best_ask = X[:, 0, 3]  # (seq_len,)
            p_mid = (best_bid + best_ask) / 2.0
            p_mid_expanded = p_mid[:, np.newaxis]  # (seq_len, 1)

        X[..., 0] = np.abs(X[..., 0] - p_mid_expanded)  # bidRate
        X[..., 3] = np.abs(X[..., 3] - p_mid_expanded)  # askRate

        X = np.log1p(X)
        # X[..., 3] = - X[..., 3]  # askRate

        return X

    def transform(self, X):
        X = self._log_transform(X)
        if hasattr(self, "mean_") and hasattr(self, "std_"):
            X = (X - self.mean_) / self.std_  # (timesteps, levels, 7) - (7,) / (7,)
        return X


class ChunkDataset(Dataset):
    """Simple dataset for pre-chunked data - no sliding windows needed."""

    def __init__(self, chunks_X, chunks_y, normalizer=None):
        """
        Args:
            chunks_X: (n_chunks, seq_len, num_levels, 7)
            chunks_y: (n_chunks,)Goal: Remove the effect of absolute price level (since midprice = $500 or $5000 doesn’t matter — only relative differences do).


            normalizer: Optional normalizer instance
        """
        self.chunks_X = chunks_X
        self.chunks_y = chunks_y
        self.normalizer = normalizer
        self.seq_len = chunks_X.shape[1]
        self.num_levels = chunks_X.shape[2]

    def __len__(self):
        return len(self.chunks_X)

    def __getitem__(self, idx):
        X = self.chunks_X[idx]  # (seq_len, num_levels, 7)

        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        else:
            X = X.copy()

        y = self.chunks_y[idx]

        return torch.from_numpy(X).float(), torch.tensor(y, dtype=torch.float32)


def get_normalized_datasets(
    data_root,
    normalization_type="none",
    train_split=0.8,
    batch_size=256,
    no_val=False,
):
    """Get train/val loaders with optional normalization."""

    # Load chunked data
    chunks_X = np.load(
        f"{data_root}/chunks_X.npy"
    )  # (n_chunks, seq_len, num_levels, 7)
    chunks_y = np.load(f"{data_root}/chunks_y.npy")  # (n_chunks,)

    print(f"Loaded chunks: X={chunks_X.shape}, y={chunks_y.shape}")

    # Split data
    if no_val:
        train_X, train_y = chunks_X, chunks_y
        val_X, val_y = None, None
    else:
        split_idx = int(train_split * len(chunks_X))
        train_X, train_y = chunks_X[:split_idx], chunks_y[:split_idx]
        val_X, val_y = chunks_X[split_idx:], chunks_y[split_idx:]

    # Select normalizer
    normalizers = {
        "none": None,
        "second": SecondNormalizer(),
    }
    normalizer = normalizers[normalization_type]

    # Fit normalizer on training data if needed
    if normalizer is not None:
        print(f"Fitting {normalization_type} normalizer on training data...")
        # Pass raw numpy arrays for fitting
        normalizer.fit(train_X)

        # Print stats if available
        if hasattr(normalizer, "stats") and "mean" in normalizer.stats:
            print(f"Normalizer stats - Mean: {normalizer.stats['mean']}")
            print(f"Normalizer stats - Std: {normalizer.stats['std']}")
        elif hasattr(normalizer, "mean_"):
            print(f"Normalizer stats - Mean: {normalizer.mean_}")
            print(f"Normalizer stats - Std: {normalizer.std_}")

    # Create datasets and loaders
    train_ds = ChunkDataset(train_X, train_y, normalizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    if no_val:
        val_loader = None
    else:
        val_ds = ChunkDataset(val_X, val_y, normalizer)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

    return train_loader, val_loader, normalizer
