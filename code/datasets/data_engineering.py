"""
Stage 1 — Data Engineering
Load Titanic raw data, clean missing values / outliers, split train/test.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET_COLUMN = "Survived"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Place titanic.csv in data/raw/."
        )
    return pd.read_csv(path)


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned = cleaned[(cleaned[column] >= lower) & (cleaned[column] <= upper)]
    return cleaned.reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Keep only useful columns for modeling
    keep = FEATURE_COLUMNS + [TARGET_COLUMN]
    data = data[keep]

    # Impute missing values
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Drop any remaining missing rows (should be rare)
    data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Remove outliers on numeric columns
    data = remove_outliers_iqr(data, ["Age", "Fare", "SibSp", "Parch"])

    return data.reset_index(drop=True)


def split_and_save(
    df: pd.DataFrame,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, Path]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def run_data_engineering() -> dict:
    raw = load_raw_data()
    cleaned = clean_data(raw)
    train_path, test_path = split_and_save(cleaned)

    summary = {
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "train_rows": int(pd.read_csv(train_path).shape[0]),
        "test_rows": int(pd.read_csv(test_path).shape[0]),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    print("Data engineering complete:", summary)
    return summary


if __name__ == "__main__":
    run_data_engineering()
