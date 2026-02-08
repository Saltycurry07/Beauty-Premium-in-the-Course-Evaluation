import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =========================
# 1) 读数据
# =========================
DATA_PATH = "faculty_scored_merged_with_rmp2_continuous_1to5_with_school.csv"
df = pd.read_csv(DATA_PATH)

# =========================
# 2) X / y
# =========================
X = df[["beauty_1to5", "avg_difficulty", "school_name"]].copy()
y = df["avg_rating"].copy()

# 丢掉 y 缺失
mask = ~y.isna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

# =========================
# 3) 预处理（基础版：beauty 标准化 + school one-hot）
# =========================
num_features = ["beauty_1to5", "avg_difficulty"]
cat_features = ["school_name"]


numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_features),
        ("cat", categorical_pipe, cat_features),
    ],
    remainder="drop",
)

# =========================
# 3.1) 可选：加入非线性（对 beauty 做二次项）
# 只对 beauty 做 PolynomialFeatures，再拼上 school one-hot
# =========================
numeric_pipe_poly = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # beauty, beauty^2
    ("scaler", StandardScaler()),
])

preprocess_poly = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe_poly, num_features),
        ("cat", categorical_pipe, cat_features),
    ],
    remainder="drop",
)

# =========================
# 4) 模型池
# =========================
models_base_preprocess = {
    "Linear": LinearRegression(),
    "Ridge(alpha=1.0)": Ridge(alpha=1.0),
    "DummyMean": DummyRegressor(strategy="mean"),

    # 你要的两种非线性模型
    "RandomForest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    ),
}

# 额外：线性模型 + beauty 二次项（看看有没有轻微提升）
models_poly_preprocess = {
    "Linear + beauty^2": LinearRegression(),
    "Ridge + beauty^2": Ridge(alpha=1.0),
}

# =========================
# 5) CV & scoring
# =========================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

def eval_models(model_dict, preprocessor, tag=""):
    rows = []
    for name, model in model_dict.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
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
            "model": (name + tag).strip(),
            "n": len(y),
            "RMSE(mean)": -out["test_rmse"].mean(),
            "MAE(mean)": -out["test_mae"].mean(),
            "R2(mean)": out["test_r2"].mean(),
        })
    return pd.DataFrame(rows)

# =========================
# 6) 跑评估并汇总输出
# =========================
res1 = eval_models(models_base_preprocess, preprocess)
res2 = eval_models(models_poly_preprocess, preprocess_poly, tag=" (poly)")

results = pd.concat([res1, res2], ignore_index=True)
results = results.sort_values(by="RMSE(mean)").reset_index(drop=True)

print("=== CV results (5-fold) ===")
print(results.to_string(index=False))
