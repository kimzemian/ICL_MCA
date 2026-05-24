import argparse
import torch
import numpy as np
from train import get_model
from preprocess import process_csv_to_chunks
import pickle
import os
import glob
from tqdm import tqdm
from normalizer import ChunkDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

def predict(model, loader, device, remainder, eval=False):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Predicting"):
            batch_X = batch_X.to(device)
            pred = model(batch_X).cpu().numpy()  # (batch_size, chunk_size)
            all_preds.append(pred)
            all_targets.append(batch_y.numpy())
    
    # Concatenate all batch predictions: (total_chunks, chunk_size)
    all_preds = np.concatenate(all_preds, axis=0)
    if remainder > 0:
        # Flatten all but last chunk, then add only 'remainder' from last chunk
        result = all_preds[:-1].flatten()
        result = np.concatenate([result, all_preds[-1, -remainder:]])
    else:
        # All chunks are full, just flatten everything
        result = all_preds.flatten()
    
    if eval:
        all_targets = np.concatenate(all_targets, axis=0)
        if remainder > 0:
            targets_flat = all_targets[:-1].flatten()
            targets_flat = np.concatenate([targets_flat, all_targets[-1, -remainder:]])
        else:
            targets_flat = all_targets.flatten()
    else:
        targets_flat = None
        

    return result, targets_flat


def ensemble_predict(models, loader, device, remainder, eval=False):
    """Average ensemble predictions using DataLoader."""
    all_model_preds = []
    for i, model in enumerate(models):
        print(f"\nPredicting with ensemble member {i+1}/{len(models)}")
        preds, targets = predict(model, loader, device=device, remainder=remainder, eval=eval)
        all_model_preds.append(preds)
    
    # Stack and average across ensemble members
    all_model_preds = np.stack(all_model_preds, axis=0)
    ensemble_preds = np.mean(all_model_preds, axis=0)

    if eval:
        r2 = r2_score(targets, ensemble_preds)
        print(f"\nEnsemble R2 Score: {r2:.6f}")
    
    return ensemble_preds


def load_ensemble_models(ensemble_dir, model_args, seq_len, device):
    checkpoint_pattern = os.path.join(ensemble_dir, "best_model_seed_*.pt")
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern))
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in: {ensemble_dir}")

    print(f"\n=== Loading {len(checkpoint_paths)} ensemble members ===")
    models = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"Loading model {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint["model_config"]
        model_args = argparse.Namespace(**config)
        model = get_model(model_args, seq_len)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        models.append(model)
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--ensemble_dir", type=str, required=True)
    parser.add_argument("--normalizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--num_levels", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate R2 if targets are available"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    chunk_size = args.chunk_size
    # === Step 1: Preprocess CSV ===
    print("\n=== Preprocessing CSV ===")
    process_csv_to_chunks(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_levels=args.num_levels,
        chunk_size=chunk_size,
    )
    X_full = np.load(f"{args.output_dir}/full_X.npy")
    L = len(X_full)
    n_full = L // chunk_size
    remainder = L % chunk_size
    
    if args.eval:
        y_full = np.load(f"{args.output_dir}/full_y.npy") 
    else:
        y_full = np.zeros((L,), dtype=np.float32)  # Dummy targets
    
    # Create array of chunks for both X and y
    if remainder > 0:
        # Reshape full chunks, keeping all dimensions
        full_chunks_X = X_full[:n_full * chunk_size].reshape(n_full, chunk_size, 8, 7)
        tail_chunk_X = X_full[-chunk_size:][np.newaxis, ...]
        chunks_X = np.concatenate([full_chunks_X, tail_chunk_X], axis=0)
        
        full_chunks_y = y_full[:n_full * chunk_size].reshape(n_full, chunk_size)
        tail_chunk_y = y_full[-chunk_size:][np.newaxis, ...]
        chunks_y = np.concatenate([full_chunks_y, tail_chunk_y], axis=0)
    else:
        # All data fits perfectly into chunks
        chunks_X = X_full.reshape(n_full, chunk_size, 8, 7)
        chunks_y = y_full.reshape(n_full, chunk_size)



    # === Step 3: Load and apply normalizer ===
    with open(args.normalizer_path, "rb") as f:
        normalizer = pickle.load(f)
    print("Normalizer loaded")

    dataset = ChunkDataset(chunks_X, chunks_y, normalizer)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # === Step 4: Load ensemble models ===
    models = load_ensemble_models(args.ensemble_dir, args, seq_len=args.chunk_size, device=device)
    print(f"✓ Loaded {len(models)} ensemble members")

    # === Step 5: Predict on full sequence ===
    print("\n=== Generating ensemble predictions ===")
    preds = ensemble_predict(models, loader, device=device, remainder=remainder, eval=args.eval)

    np.save(f"{args.output_dir}/predictions.npy", preds)
    print(f"\nSaved {len(preds)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
