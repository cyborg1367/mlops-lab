from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    # Load the Breast Wisconsin dataset
    data = load_breast_cancer(as_frame=True)

    df = data.frame.copy()
    df["target"] = (data.target == 0).astype(int)

    # Create outout directory
    output_dir = Path("data")
    output_dir.mkdir(parents = True, exist_ok = True)

    # Save raw dataset
    output_path = output_dir / "raw.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())

if __name__ == "__main__":
    main()