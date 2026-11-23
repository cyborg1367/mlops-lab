from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input_path = Path("data/raw.csv")
    output_dir = Path("data/processed")

    output_dir.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 100, stratify = df["target"])

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index = False)
    test_df.to_csv(test_path, index = False)

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nTrain target distribution:")
    print(train_df["target"].value_counts(normalize=True))

    print("\nTest target distribution:")
    print(test_df["target"].value_counts(normalize=True))


if __name__ == "__main__":
    main()