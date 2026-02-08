# model_difficulty_only.py

import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor


# =========================
# 0) 环境自检输出（你要的“环境里需要输出什么”）
# =========================
def print_env_info():
    import sklearn
    print("=== Environment check ===")
    print("Python executable:", sys.executable)
    print("Python version   :", sys.version.split()[0])
    print("pandas version   :", pd.__version__)
    print("numpy version    :", np.__version__)
    print("sklearn version  :", sklearn.__version__)
    print("=========================\n")


# =========================
# 1) 读数据
# =========================
DATA_PATH = "faculty_scored_merged_with_rmp2_continuous_1to5_with_school.csv"


def main():
    print_env_info()

    df = pd.read_csv(DATA_PATH)

    # 数据列名检查（避免 KeyError）
    print("=== Columns preview ===")
    print("Total columns:", len(df.columns))
    print("First 30 columns:", df.columns.tolist()[:30])
    print("=======================\n")

    required_cols = ["avg_difficulty", "avg_rating"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =========================
    # 2) X / y（只用 difficulty）
    # =========================
    X = df[["avg_difficulty"]].copy()
    y = df["avg_rating"].copy()

    # 丢掉 y 缺失；同时也把 X 对齐
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    print("=== Data preview ===")
    print("n (after dropping missing y):", len(y))
    print("y (avg_rating) mean/std:", float(y.mean()), float(y.std()))
    print("X (avg_difficulty) missing count:", int(X["avg_difficulty"].isna().sum()))
    print("====================\n")

    # =========================
    # 3) 预处理（数值：补缺失 + 标准化）
    # =========================
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), ["avg_difficulty"]),
        ],
        remainder="drop",
    )

    # =========================
    # 4) 模型（Linear / Ridge / DummyMean）
    # =========================
    models = {
        "Linear": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "DummyMean": DummyRegressor(strategy="mean"),
    }

    # =========================
    # 5) 5-fold CV + 指标
    # =========================
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model),
        ])

        out = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        rows.append({
            "model": name,
            "n": len(y),
            "RMSE(mean)": -out["test_rmse"].mean(),
            "MAE(mean)": -out["test_mae"].mean(),
            "R2(mean)": out["test_r2"].mean(),
        })

    results = pd.DataFrame(rows).sort_values(by="RMSE(mean)").reset_index(drop=True)

    print("=== CV results (5-fold) ===")
    print(results.to_string(index=False))
    print("===========================")


if __name__ == "__main__":
    main()
