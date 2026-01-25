from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def main():
    # ========= 0) 路径基准：当前脚本所在文件夹 =========
    ROOT = Path(__file__).resolve().parent
    print("ROOT:", ROOT)

    # ========= 1) 模型文件（必须在同一文件夹） =========
    prototxt = ROOT / "resnet18_deploy.prototxt"
    caffemodel = ROOT / "resnet18.caffemodel"

    if not prototxt.exists():
        raise FileNotFoundError(f"Missing prototxt: {prototxt}")
    if not caffemodel.exists():
        raise FileNotFoundError(f"Missing caffemodel: {caffemodel}")

    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    print("Model loaded OK")

    # ========= 2) 输入 CSV（你现在这个） =========
    csv_in = ROOT / "nyu_cs_faculty.csv"
    if not csv_in.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_in}")

    df = pd.read_csv(csv_in)

    # 必须包含这两列
    required = {"name", "local_path"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {sorted(required)}\n"
            f"Your columns: {list(df.columns)}"
        )

    # ========= 3) 逐张图片预测 =========
    scores = []
    missing = 0
    read_fail = 0

    for rel in tqdm(df["local_path"].astype(str), desc="Predicting"):
        img_path = ROOT / rel  # local_path 是文件名或相对路径

        if not img_path.exists():
            scores.append(None)
            missing += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            scores.append(None)
            read_fail += 1
            continue

        # resize + blob
        img = cv2.resize(img, (224, 224))
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0,
            size=(224, 224),
            mean=(104, 117, 123),
            swapRB=False,
        )

        net.setInput(blob)
        out = net.forward()
        score_raw = float(out.squeeze())
        scores.append(score_raw)

    df["pred_score_raw"] = scores
    df["pred_score_1_5"] = df["pred_score_raw"].clip(1.0, 5.0)

    # ========= 4) 输出 =========
    out_csv = ROOT / "nyu_cs_faculty_scored.csv"
    df.to_csv(out_csv, index=False)

    print("\nDONE")
    print("Missing images:", missing)
    print("Read fail:", read_fail)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()