import os
import sys
import glob
import shutil
import subprocess
import pandas as pd

# ================== 你只需要改这三项 ==================
CSV_PATH = r"./faculty_scored_merged_with_rmp2.8_reassigned_by_school_gender_rank.csv"
IMAGE_BASE_DIR = r"."  # 如果 local_path 已经是绝对路径，可留 "" 或 "."
SHAPE_PREDICTOR = r"./shape_predictor_68_face_landmarks.dat"
# ======================================================

OUT_DIR = "./avg_outputs"
WORK_DIR = "./_avg_work"

BINS = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0000001)]
BIN_LABELS = ["1-2", "2-3", "3-4", "4-5"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def normalize_gender(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["m", "male", "man", "1"]:
        return "M"
    if s in ["f", "female", "woman", "0", "2"]:
        return "F"
    return None


def bin_beauty(v):
    if pd.isna(v):
        return None
    try:
        v = float(v)
    except Exception:
        return None
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        if v >= lo and v < hi:
            return lab
    return None


def resolve_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(IMAGE_BASE_DIR, p))


def run_cmd(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=cwd)


def clean_images_without_landmarks(group_dir: str, min_keep: int = 5) -> int:
    imgs = [p for p in glob.glob(os.path.join(group_dir, "*")) if p.lower().endswith(IMG_EXTS)]

    kept = 0
    removed = 0
    for img_path in imgs:
        txt_path = img_path + ".txt"
        if os.path.exists(txt_path):
            kept += 1
        else:
            try:
                os.remove(img_path)
            except Exception:
                pass
            removed += 1

    print(f"[CLEAN] {os.path.basename(group_dir)} kept={kept}, removed_no_landmarks={removed}")
    if kept < min_keep:
        print(f"[SKIP] {os.path.basename(group_dir)}: too few valid faces after cleaning (kept={kept})")
    return kept


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required = {"local_path", "beauty_1to5", "gender_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺列: {missing}. 你现在的列名是: {list(df.columns)}")

    df["gender_norm"] = df["gender_pred"].apply(normalize_gender)
    df["beauty_bin"] = df["beauty_1to5"].apply(bin_beauty)
    df = df.dropna(subset=["gender_norm", "beauty_bin", "local_path"]).copy()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for g in ["M", "F"]:
        for b in BIN_LABELS:
            tag = f"{'male' if g == 'M' else 'female'}_{b}"
            group_dir = os.path.join(WORK_DIR, tag)
            group_dir_abs = os.path.abspath(group_dir)

            group = df[(df["gender_norm"] == g) & (df["beauty_bin"] == b)]
            paths = group["local_path"].tolist()

            if os.path.exists(group_dir_abs):
                shutil.rmtree(group_dir_abs)
            os.makedirs(group_dir_abs, exist_ok=True)

            # copy images into group folder as 00000.xxx
            copied = 0
            for p in paths:
                src = resolve_path(p)
                if not os.path.exists(src):
                    continue
                ext = os.path.splitext(src)[1].lower()
                if ext not in IMG_EXTS:
                    continue
                dst = os.path.join(group_dir_abs, f"{copied:05d}{ext}")
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    continue

            print(f"\n=== Group {tag}: copied {copied} images ===")
            if copied < 5:
                print(f"[SKIP] {tag}: not enough images")
                continue

            # 1) extract landmarks
            run_cmd([sys.executable, "extract.py", SHAPE_PREDICTOR, group_dir_abs], cwd=script_dir)

            # 2) remove images without txt
            kept = clean_images_without_landmarks(group_dir_abs, min_keep=5)
            if kept < 5:
                continue

            # 3) average face (writes group_dir_abs/average_face.jpg)
            run_cmd([sys.executable, os.path.join(script_dir, "average.py"), group_dir_abs, "400", "400"], cwd=group_dir_abs)

            avg_src = os.path.join(group_dir_abs, "average_face.jpg")
            if not os.path.exists(avg_src):
                raise RuntimeError(f"average.py did not produce output: {avg_src}")

            out_path = os.path.join(OUT_DIR, f"avg_{tag}.jpg")
            shutil.copy2(avg_src, out_path)
            print(f"[OK] Saved: {out_path}")

    print("\nDone. Outputs in:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
