"""
eda.py
------
Exploratory Data Analysis for the Heart Disease UCI dataset.
Produces the exact visualization set the FAQ names explicitly:
  - histograms
  - correlation heatmap
  - class distribution plot
  - missing value analysis
  - feature relationship analysis

Run: python src/eda.py
Output: reports/figures/*.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — this runs in CI too, not just interactively
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import NUMERIC_FEATURES, clean_and_binarize

DATA_PATH = Path(__file__).parent.parent / "data" / "heart_disease.csv"
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_raw_for_eda() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#4C72B0")
        ax.set_xlabel("Missing count")
        ax.set_title("Missing values by column (raw data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missing_values.png", dpi=150)
    plt.close(fig)
    print(f"[eda] missing values:\n{missing if len(missing) else 'none'}")


def plot_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    for ax in axes.flat[len(NUMERIC_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Distributions of numeric features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "histograms.png", dpi=150)
    plt.close(fig)


def plot_class_distribution(df_clean: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df_clean["target"].value_counts().sort_index()
    labels = counts.index.map({0: "No disease", 1: "Disease"})
    sns.barplot(x=labels, y=counts.values, hue=labels, legend=False,
                ax=ax, palette=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    ax.set_ylabel("Patient count")
    ax.set_title(f"Class balance (n={len(df_clean)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[eda] class balance:\n{counts}")


def plot_correlation_heatmap(df_clean: pd.DataFrame) -> None:
    corr = df_clean[NUMERIC_FEATURES + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (numeric features + target)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_feature_relationships(df_clean: pd.DataFrame) -> None:
    # Numeric-vs-target: boxplots
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(20, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df_clean, x="target", y=col, hue="target", legend=False,
                    ax=ax, palette=["#4C72B0", "#C44E52"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No disease", "Disease"])
        ax.set_title(col)
    fig.suptitle("Numeric feature vs. target")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_relationships_numeric.png", dpi=150)
    plt.close(fig)

    # Categorical-vs-target: chest pain type is the most clinically meaningful one
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_clean, x="cp", hue="target", ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title("Chest pain type (cp) vs. target")
    ax.legend(title="target", labels=["No disease", "Disease"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_relationships_categorical.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df_raw = load_raw_for_eda()
    df_clean = clean_and_binarize(df_raw.copy())

    plot_missing_values(df_raw)
    plot_histograms(df_clean)
    plot_class_distribution(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_feature_relationships(df_clean)

    print(f"[eda] saved {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
