"""Audit the patch classification splits for train/test leakage.

Replays the exact splits made by src/PatchDataset.py and reports how much of
the test set is visible during training. Needs only pandas and scikit-learn,
and reads the manifests in splits/, so it runs on a fresh clone.

    python scripts/audit_splits.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = Path(__file__).resolve().parent.parent / "splits"
SEED = 42  # same seed and 80/20 ratio as load_dataset() / load_augmented_dataset()


def keys(df):
    """Patch identity. ids restart at 0 per class, so the type is part of the key."""
    return set(zip(df["patch_id"].astype(str), df["type"]))


def split(df):
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])


def report_augmented(name, df, test):
    train, val = split(df)
    is_aug = train["patch_id"].astype(str).str.contains("aug")
    originals, augmented = train[~is_aug], train[is_aug]
    sources = set(zip(
        augmented["patch_id"].astype(str).str.extract(r"patch(\d+)_aug")[0],
        augmented["type"].str.replace("_aug", ""),
    ))
    verbatim = test & keys(originals)
    seen = verbatim | (test & sources)
    print(f"\n{name}: {len(train):,} train / {len(val):,} val")
    print(f"  test patches in training, verbatim:       {len(verbatim):>5,} ({len(verbatim) / len(test):.1%})")
    print(f"  test patches in training, in any form:    {len(seen):>5,} ({len(seen) / len(test):.1%})")


def main():
    patches = pd.read_csv(SPLITS / "v3_all_patches.csv")
    test_df = pd.read_csv(SPLITS / "v3_test.csv")
    test = keys(test_df)

    train_pool = patches[[key not in test for key in zip(patches["patch_id"].astype(str), patches["type"])]]
    shared = set(train_pool["filename"]) & set(test_df["filename"])
    print(f"V3: {len(patches):,} patches from {patches['filename'].nunique()} images, {len(test):,} in test")
    print(f"  source images in both train and test:     {len(shared)} of {test_df['filename'].nunique()}")

    train, val = split(patches)
    print(f"\nV3 base: {len(train):,} train / {len(val):,} val")
    print(f"  test patches in training:                 {len(test & keys(train)):>5,}")
    print(f"  test patches in the validation split:     {len(test & keys(val)):>5,}  (validation == test)")

    report_augmented("V3 + aug", pd.read_csv(SPLITS / "v3_aug_train.csv"), test)
    report_augmented("V4", pd.read_csv(SPLITS / "v4_aug_train.csv"), test)


if __name__ == "__main__":
    main()
