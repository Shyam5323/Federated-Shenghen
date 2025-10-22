from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_shenzhen_dataframe(image_root: Path) -> pd.DataFrame:
    """Build a dataframe containing absolute paths, labels, and patient ids."""
    records = []
    for image_path in sorted(image_root.glob('*.png')):
        stem_parts = image_path.stem.split('_')
        if len(stem_parts) < 3:
            continue

        label = int(stem_parts[-1])
        patient_id = stem_parts[1]

        records.append(
            {
                'path': str(image_path.resolve()),
                'label': label,
                'patient_id': patient_id,
            }
        )

    if not records:
        raise FileNotFoundError(f"No PNG images found in '{image_root}'.")

    return pd.DataFrame(records)


def report_and_save_partitions(partitions: dict, strategy_name: str, output_root: Path) -> None:
    """Emit summary statistics and persist partitions to CSV files."""
    print(f"\n--- Reporting for Partition Strategy: {strategy_name} ---")
    output_dir = output_root / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id, data in partitions.items():
        client_df = data.reset_index(drop=True)
        client_name = f'client_{client_id}'
        client_file = output_dir / f'{client_name}.csv'
        images_file = output_dir / f'{client_name}_images.csv'

        client_df.to_csv(client_file, index=False)
        client_df[['path', 'label']].to_csv(images_file, index=False)

        label_counts = client_df['label'].value_counts().sort_index()
        unique_patients = client_df['patient_id'].nunique()

        print(f"\nClient {client_id}:")
        print(f"  Total images: {len(client_df)}")
        print(f"  Saved to: {client_file}")
        if not client_df.empty:
            print("  Label distribution (0=no TB, 1=TB):")
            for label, count in label_counts.items():
                pct = (count / len(client_df)) * 100 if len(client_df) else 0
                print(f"    Label {label}: {count} ({pct:.2f}%)")
            print(f"  Unique patients: {unique_patients}")

    print("\n" + "=" * 60 + "\n")


def partition_iid(df: pd.DataFrame, num_clients: int, seed: int) -> dict:
    """IID split across clients."""
    print("\nPartitioning data using IID strategy...")
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    partitions = np.array_split(df_shuffled, num_clients)
    return {i: partitions[i] for i in range(num_clients)}


def partition_by_patient_block(df: pd.DataFrame, num_clients: int, seed: int) -> dict:
    """Assign whole patient cohorts to individual clients (non-IID)."""
    print("\nPartitioning data by patient blocks (Pathological Non-IID)...")
    patient_ids = df['patient_id'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    patient_splits = np.array_split(patient_ids, num_clients)

    partitions = {}
    for idx, patient_subset in enumerate(patient_splits):
        mask = df['patient_id'].isin(patient_subset)
        partitions[idx] = df[mask].reset_index(drop=True)

    return partitions


def partition_by_label_skew(df: pd.DataFrame, num_clients: int, alpha: float, seed: int) -> dict:
    """Dirichlet-based label skew across clients."""
    print(f"\nPartitioning data by label skew (Dirichlet with alpha={alpha})...")
    rng = np.random.default_rng(seed)
    partitions = {i: [] for i in range(num_clients)}

    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label].sample(frac=1, random_state=seed + int(label)).reset_index(drop=True)
        probs = rng.dirichlet(np.repeat(alpha, num_clients))
        counts = rng.multinomial(len(label_df), probs)

        start = 0
        for client_idx, count in enumerate(counts):
            end = start + count
            if count > 0:
                partitions[client_idx].append(label_df.iloc[start:end])
            start = end

    concatenated = {}
    for client_idx, frames in partitions.items():
        if frames:
            concatenated[client_idx] = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            concatenated[client_idx] = pd.DataFrame(columns=df.columns)

    return concatenated


if __name__ == '__main__':
    NUM_CLIENTS = 7
    LABEL_SKEW_ALPHA = 0.5
    SEED = 42
    TEST_RATIO = 0.15
    VAL_RATIO = 0.15

    dataset_dir = Path(__file__).resolve().parents[1] / 'Shenzhen-CXR-PNGs'
    partitions_root = Path(__file__).resolve().parent / 'partitions'
    partitions_root.mkdir(exist_ok=True)

    print(f"Loading Shenzhen dataset from {dataset_dir}...")
    full_df = load_shenzhen_dataframe(dataset_dir)

    # Split into train + temp/test first
    train_val_df, test_df = train_test_split(
        full_df,
        test_size=TEST_RATIO,
        stratify=full_df['label'],
        random_state=SEED,
    )

    # Derive validation ratio relative to remaining data
    val_relative = VAL_RATIO / (1 - TEST_RATIO)
    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=train_val_df['label'],
        random_state=SEED,
    )

    # Persist global splits for convenience
    train_df.to_csv(partitions_root / 'global_train.csv', index=False)
    valid_df.to_csv(partitions_root / 'global_valid.csv', index=False)
    test_df.to_csv(partitions_root / 'global_test.csv', index=False)

    print(
        "Training images: {train} | Validation images: {valid} | Test images: {test}".format(
            train=len(train_df),
            valid=len(valid_df),
            test=len(test_df),
        )
    )

    iid_partitions = partition_iid(train_df, NUM_CLIENTS, seed=SEED)
    report_and_save_partitions(iid_partitions, 'iid', partitions_root)

    pathological_partitions = partition_by_patient_block(train_df, NUM_CLIENTS, seed=SEED)
    report_and_save_partitions(pathological_partitions, 'pathological_non_iid', partitions_root)

    label_skew_partitions = partition_by_label_skew(train_df, NUM_CLIENTS, alpha=LABEL_SKEW_ALPHA, seed=SEED)
    report_and_save_partitions(label_skew_partitions, 'label_skew', partitions_root)
