from pathlib import Path

import pandas as pd
import yaml

from sklearn.model_selection import train_test_split


def main():
    # Load configuration
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = config["data"]

    # Load raw dataset
    input_path = Path("data/raw.csv")
    output_dir = Path("data/processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a target column.")

    # Split into development and test sets
    development_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["target"],
    )

    # Save datasets
    development_path = output_dir / "development.csv"
    test_path = output_dir / "test.csv"

    development_df.to_csv(development_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Report split information
    print(f"Total samples: {len(df)}")
    print(f"Development samples: {len(development_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nDevelopment target distribution:")
    print(development_df["target"].value_counts(normalize=True))

    print("\nTest target distribution:")
    print(test_df["target"].value_counts(normalize=True))

    print(f"\nDevelopment data saved to: {development_path}")
    print(f"Test data saved to: {test_path}")


if __name__ == "__main__":
    main()
