import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42
CSV_PATH = "cars.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# Load + basic engineering
# =========================
df = pd.read_csv(CSV_PATH)

if "Year" in df.columns:
    df["Vehicle_Age"] = 2025 - df["Year"]
    df = df.drop(columns=["Year"])

# =========================
# 1) Histogram – Selling Price
# =========================
if "Selling_Price" in df.columns:
    plt.figure()
    df["Selling_Price"].hist(bins=20)
    plt.xlabel("Selling Price")
    plt.ylabel("Count")
    plt.title("Histogram of Selling Price")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hist_selling_price.png", dpi=300)
    plt.close()

# =========================
# 2) Scatter – Present Price vs Selling Price
# =========================
if {"Present_Price", "Selling_Price"}.issubset(df.columns):
    plt.figure()
    plt.scatter(df["Present_Price"], df["Selling_Price"], alpha=0.6)
    plt.xlabel("Present Price")
    plt.ylabel("Selling Price")
    plt.title("Present Price vs Selling Price")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scatter_present_vs_selling.png", dpi=300)
    plt.close()

# =========================
# 3) Correlation Heatmap (numeric features)
# =========================
numeric = df.select_dtypes(include=[np.number])
if numeric.shape[1] > 1:
    corr = numeric.corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    cols = corr.columns
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)

    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )

    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "corr_heatmap.png", dpi=300)
    plt.close()

# =========================
# 4) Confusion Matrix – Logistic Regression
# =========================
if "Transmission" in df.columns:
    tmp = df.copy()
    tmp["y_cls"] = (
        tmp["Transmission"]
        .astype(str)
        .str.lower()
        .str.contains("auto")
        .astype(int)
    )

    X_cls_raw = tmp.drop(columns=["y_cls", "Transmission"])
    X_cls = pd.get_dummies(X_cls_raw, drop_first=True)
    y_cls = tmp["y_cls"].values

    Xctr, Xcte, yctr, ycte = train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    cls_model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    )
    cls_model.fit(Xctr, yctr)
    ypred = cls_model.predict(Xcte)

    cm = confusion_matrix(ycte, ypred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Manual", "Automatic"])
    ax.set_yticklabels(["Manual", "Automatic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Automatic vs Manual")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix_logistic.png", dpi=300)
    plt.close()
