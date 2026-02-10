import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

RANDOM_STATE = 42
CSV_PATH = "cars.csv"


df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print(df.head())


if "Year" in df.columns:
    df["Vehicle_Age"] = 2025 - df["Year"]
    df = df.drop(columns=["Year"])


if "Selling_Price" in df.columns:
    y_reg = df["Selling_Price"].values
    X_reg_raw = df.drop(columns=["Selling_Price"])

    X_reg = pd.get_dummies(X_reg_raw, drop_first=True)

    Xtr, Xte, ytr, yte = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    reg_model = make_pipeline(
        StandardScaler(with_mean=False),
        LinearRegression()
    )

    reg_model.fit(Xtr, ytr)
    yhat = reg_model.predict(Xte)

    mae = mean_absolute_error(yte, yhat)
    mse = mean_squared_error(yte, yhat)
    rmse = np.sqrt(mse)
    r2 = r2_score(yte, yhat)

    print("\n=== Linear Regression ===")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

    lin_step = reg_model.named_steps["linearregression"]
    coef_series = pd.Series(
        lin_step.coef_,
        index=X_reg.columns
    ).sort_values(key=np.abs, ascending=False).head(12)

    print("\nTop Coefficients:")
    print(coef_series)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        reg_model,
        X_reg,
        y_reg,
        scoring="neg_mean_absolute_error",
        cv=cv,
    )

    print("\n5-Fold CV (MAE):", -cv_scores.mean(), "STD:", cv_scores.std())


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

    acc = accuracy_score(ycte, ypred)
    prec = precision_score(ycte, ypred)
    rec = recall_score(ycte, ypred)
    cm = confusion_matrix(ycte, ypred)

    print("\n=== Logistic Regression (Automatic vs Manual) ===")
    print("Accuracy        :", acc)
    print("Precision (Auto):", prec)
    print("Recall (Auto)   :", rec)
    print("Confusion Matrix:\n", cm)
