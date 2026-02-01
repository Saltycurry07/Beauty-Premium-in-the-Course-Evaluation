import os
import cv2
import numpy as np
import pandas as pd

# ========= 你只需要保证这些文件在当前文件夹 =========
CSV_IN  = r"./facluty_photo.csv"          # <-- 改成你的实际 csv 文件名
IMG_DIR = r"."                           # 照片就在当前目录
PROTO   = r"./resnext50_deploy.prototxt"
MODEL   = r"./resnext50.caffemodel"

CSV_OUT = r"./MIT_UCLA_USC_scored.csv"   # 输出

# ===== Caffe 预处理 =====
INPUT_SIZE = (224, 224)
MEAN_BGR = (104, 117, 123)
SCALE = 1.0
SWAP_RB = False

# ===== 分学校映射配置 =====
# 用分位数裁剪避免极端值影响（推荐）
Q_LOW  = 0.01
Q_HIGH = 0.99


def pick_school_col(df: pd.DataFrame) -> str:
    """
    自动找“学校名”那一列。
    你说它在从左往右第2列，但列名可能不同（school_name / school / school_nar 等）
    """
    # 1) 优先用常见列名
    candidates = ["school_name", "school", "school_nar", "school_na", "university", "campus"]
    for c in candidates:
        if c in df.columns:
            return c

    # 2) 兜底：按“第二列”
    if df.shape[1] >= 2:
        return df.columns[1]

    raise ValueError("CSV has fewer than 2 columns; cannot locate school column.")


def score_one_image(net, img_path: str) -> float:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=SCALE,
        size=INPUT_SIZE,
        mean=MEAN_BGR,
        swapRB=SWAP_RB,
        crop=False
    )
    net.setInput(blob)
    out = net.forward()  # (1,1)
    return float(out.reshape(-1)[0])


def main():
    # 基础检查
    for p, kind in [(CSV_IN, "CSV_IN"), (PROTO, "PROTO"), (MODEL, "MODEL")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{kind} not found: {p}")
    if not os.path.isdir(IMG_DIR):
        raise NotADirectoryError(f"IMG_DIR not found: {IMG_DIR}")

    df = pd.read_csv(CSV_IN)

    # 必要列：local_path
    if "local_path" not in df.columns:
        raise ValueError(f"CSV must contain column 'local_path'. Found columns: {list(df.columns)}")

    school_col = pick_school_col(df)
    print(f"Using school column: {school_col}")

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

    raw_scores = []
    status = []

    n = len(df)
    print(f"Scoring {n} images...")

    for i, row in df.iterrows():
        rel = str(row["local_path"]).strip()
        img_path = os.path.join(IMG_DIR, rel)

        if not os.path.exists(img_path):
            raw_scores.append(np.nan)
            status.append("missing")
            continue

        try:
            s = score_one_image(net, img_path)
            raw_scores.append(s)
            status.append("ok")
        except Exception:
            raw_scores.append(np.nan)
            status.append("error")

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"Processed {i+1}/{n}")

    df["beauty_raw"] = raw_scores
    df["status"] = status
    df["beauty_1to5"] = np.nan

    # ===== 核心：按学校分组分别计算 1–5 =====
    ok = df["status"] == "ok"

    # 注意：可能有同校样本非常少（比如 1-2 个），要做边界处理
    for school, sub_idx in df.loc[ok].groupby(school_col).groups.items():
        s = df.loc[sub_idx, "beauty_raw"].astype(float)

        # 若同校有效样本太少，无法稳定做分位数/映射：给一个常数 3.0 或退化 min-max
        if len(s) < 3:
            # 退化策略：都给 3（中性分），并在 status 里可选标注
            df.loc[sub_idx, "beauty_1to5"] = 3.0
            continue

        lo = s.quantile(Q_LOW)
        hi = s.quantile(Q_HIGH)

        # 防止 hi == lo（所有分数几乎一样）
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            df.loc[sub_idx, "beauty_1to5"] = 3.0
            continue

        s_clip = s.clip(lo, hi)
        df.loc[sub_idx, "beauty_1to5"] = 1 + 4 * (s_clip - lo) / (hi - lo)

    df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print("Saved:", os.path.abspath(CSV_OUT))
    print("\nPer-school 1–5 stats (ok only):")
    print(df.loc[ok].groupby(school_col)["beauty_1to5"].describe())


if __name__ == "__main__":
    main()
