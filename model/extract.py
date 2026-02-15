import os
import sys
import glob
import traceback
import numpy as np
import cv2
import dlib


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_unicode_color(path: str):
    """Windows 中文路径安全读取：np.fromfile + cv2.imdecode"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR uint8
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, img_bgr_uint8: np.ndarray, quality: int = 95):
    """Windows 中文路径安全写入：cv2.imencode + tofile"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpeg"]:
        ext = ".jpg"
        path = os.path.splitext(path)[0] + ".jpg"
    if ext not in [".jpg", ".png", ".bmp", ".webp"]:
        ext = ".jpg"
        path = path + ".jpg"

    params = []
    if ext == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    ok, buf = cv2.imencode(ext, img_bgr_uint8, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(path)
    return path


def safe_reencode_inplace(img_path: str, max_side: int = 2000) -> str:
    """
    把图先读进来（unicode-safe），必要时 resize，然后重新保存成标准 8-bit JPG。
    这样可以避免 dlib 因为奇怪格式/坏文件/超大图而直接 native crash。
    返回新的（可能同名）图片路径。失败则返回原路径。
    """
    img = imread_unicode_color(img_path)
    if img is None:
        return img_path

    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 强制输出为 jpg（dlib 最稳）
    base = os.path.splitext(img_path)[0]
    jpg_path = base + ".jpg"

    try:
        out = imwrite_unicode(jpg_path, img, quality=95)
        # 如果原文件不是 .jpg，就删掉原文件，避免后续混乱（可选）
        if os.path.abspath(out) != os.path.abspath(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass
        return out
    except Exception:
        return img_path


def write_landmarks(txt_path: str, shape):
    with open(txt_path, "w", encoding="utf-8") as f:
        for i in range(68):
            p = shape.part(i)
            f.write(f"{p.x} {p.y}\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract.py <shape_predictor_68_face_landmarks.dat> <images_folder>")
        sys.exit(1)

    predictor_path = sys.argv[1]
    images_folder = sys.argv[2]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # collect images
    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(images_folder, f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[WARN] No images found in: {images_folder}")
        return

    skipped = 0
    ok = 0

    for img_path in image_paths:
        try:
            # ---- 关键：先重编码（防 dlib native crash）----
            img_path2 = safe_reencode_inplace(img_path)

            print(f"Processing file: {img_path2}")

            img = imread_unicode_color(img_path2)
            if img is None:
                print("  [SKIP] unreadable image")
                skipped += 1
                continue

            # dlib wants RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            dets = detector(rgb, 1)
            print(f"Number of faces detected: {len(dets)}")
            if len(dets) == 0:
                skipped += 1
                continue

            # 用第一张脸
            d = dets[0]
            shape = predictor(rgb, d)

            txt_path = img_path2 + ".txt"
            write_landmarks(txt_path, shape)

            ok += 1

        except Exception as e:
            # Python-level errors can be caught; native crash cannot be caught.
            print(f"  [ERROR] {e}")
            print(traceback.format_exc())
            skipped += 1
            continue

    print(f"[DONE] ok={ok}, skipped={skipped}")


if __name__ == "__main__":
    main()
