import os
from collections import Counter

import pandas as pd


def parse_label_from_filename(filename: str) -> int:
    """Infer the tuberculosis label from a Shenzhen filename.

    Filenames follow the pattern ``CHNCXR_<patient_id>_<label>.png`` where the
    trailing ``_0`` indicates a normal scan and ``_1`` indicates TB.
    """
    stem = os.path.splitext(filename)[0]
    try:
        return int(stem.split('_')[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Unable to infer label from '{filename}'. Expected trailing '_0' or '_1'."
        ) from exc


def build_shenzhen_dataframe(image_dir: str) -> pd.DataFrame:
    """Return a dataframe with absolute paths, labels, and patient ids."""
    records = []
    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith('.png'):
            continue

        label = parse_label_from_filename(filename)
        patient_id = filename.split('_')[1] if '_' in filename else 'unknown'
        full_path = os.path.join(image_dir, filename)

        records.append(
            {
                'path': full_path,
                'label': label,
                'patient_id': patient_id,
                'filename': filename,
            }
        )

    if not records:
        raise FileNotFoundError(f"No PNG images found in '{image_dir}'.")

    return pd.DataFrame(records)


def analyze_shenzhen_dataset(image_dir: str) -> None:
    """Print basic statistics for the Shenzhen dataset."""
    print(f"--- Analyzing Shenzhen dataset at: {image_dir} ---")
    df = build_shenzhen_dataframe(image_dir)

    label_counts = df['label'].value_counts().sort_index()
    total_images = len(df)
    patient_counts = df.groupby('label')['patient_id'].nunique().sort_index()

    print(f"Total images: {total_images}")
    print("Label distribution (0 = no TB, 1 = TB):")
    for label, count in label_counts.items():
        pct = (count / total_images) * 100
        patients = patient_counts.get(label, 0)
        print(f"  Label {label}: {count} images ({pct:.2f}%), {patients} unique patients")

    print("\nTop 5 patients by number of images:")
    top_patients = Counter(df['patient_id']).most_common(5)
    for patient_id, count in top_patients:
        print(f"  Patient {patient_id}: {count} images")

    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Shenzhen-CXR-PNGs'))
    analyze_shenzhen_dataset(base_dir)

