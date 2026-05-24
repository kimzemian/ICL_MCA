import argparse
import polars as pl
import numpy as np
import os


def process_csv_to_chunks(csv_path, output_dir, num_levels=8, chunk_size=64):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {csv_path} with Polars...")

    # Read lazily, treating "NaN" and empty as nulls
    df = pl.scan_csv(
        csv_path,
        null_values=["NaN", ""],
        infer_schema_length=10000,  # check more rows to infer types
        ignore_errors=True,         # skip any weird malformed rows
    )

    # Generate column groups
    rate_cols = [f"{s}Rate_{i}" for s in ["bid", "ask"] for i in range(num_levels)]
    size_nc_cols = [f"{s}{t}_{i}" for s in ["bid", "ask"] for i in range(num_levels) for t in ["Size", "Nc"]]

    # --- Fill NaNs efficiently ---
    # Force all columns to Float64 to handle NaNs cleanly
    for col in rate_cols + size_nc_cols:
        df = df.with_columns(pl.col(col).cast(pl.Float64))

    # Forward fill rates
    for col in rate_cols:
        df = df.with_columns(pl.col(col).forward_fill())

    # Fill size/Nc with 0 (after cast)
    for col in size_nc_cols:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    # Materialize (this triggers actual computation)
    df = df.collect()

    # --- Convert to NumPy arrays ---
    def stack_cols(prefix):
        return np.stack([df[f"{prefix}_{i}"].to_numpy() for i in range(num_levels)], axis=1)

    bid_rates = stack_cols("bidRate")
    ask_rates = stack_cols("askRate")
    bid_sizes = stack_cols("bidSize")
    ask_sizes = stack_cols("askSize")
    bid_ncs = stack_cols("bidNc")
    ask_ncs = stack_cols("askNc")

    mids = ((bid_rates + ask_rates) / 2)[..., None]  # shape (n, num_levels, 1)

    X = np.stack(
        [bid_rates, bid_sizes, bid_ncs, ask_rates, ask_sizes, ask_ncs],
        axis=-1,  # (n, num_levels, 6)
    )
    X = np.concatenate([X, mids], axis=-1).astype(np.float32)  # (n, num_levels, 7)
    np.save(os.path.join(output_dir, "full_X.npy"), X)

    # --- Chunking ---
    n = len(X) // chunk_size
    X = X[: n * chunk_size].reshape(n, chunk_size, num_levels, 7)
    print(f"X chunked to {X.shape}")

    # --- y if present ---
    if "y" in df.columns:
        y = df["y"].to_numpy().astype(np.float32)
        np.save(os.path.join(output_dir, "full_y.npy"), y)
        y = y[: n * chunk_size].reshape(n, chunk_size)
        np.save(os.path.join(output_dir, "chunks_y.npy"), y)
        print(f"Saved y: {y.shape}")


    np.save(os.path.join(output_dir, "chunks_X.npy"), X)
    print(f"Saved X: {X.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_levels", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=64)
    a = p.parse_args()
    process_csv_to_chunks(a.csv_path, a.output_dir, a.num_levels, a.chunk_size)
