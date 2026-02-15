#!/usr/bin/env python
# coding: utf-8

import os
import math
import sys
import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_unicode_color(path: str):
    """
    Windows 中文路径安全读取：np.fromfile + cv2.imdecode
    返回 BGR uint8 或 None
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, img_bgr_uint8: np.ndarray):
    """
    Windows 中文路径安全写入：cv2.imencode + tofile
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        ext = ".jpg"
        path = path + ext

    # choose encoder ext
    encode_ext = ext
    if encode_ext == ".jpeg":
        encode_ext = ".jpg"

    ok, buf = cv2.imencode(encode_ext, img_bgr_uint8)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for: {path}")
    buf.tofile(path)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python average.py <folder> [w] [h]\n"
            "The folder should contain images and landmark .txt files.\n"
        )
        sys.exit(1)

    # Default size
    w = 170
    h = 240

    # Optional override
    if len(sys.argv) >= 4:
        if str(sys.argv[2]).isdigit():
            w = int(sys.argv[2])
        if str(sys.argv[3]).isdigit():
            h = int(sys.argv[3])

    folder = os.path.abspath(sys.argv[1])
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Load paired (image, points) safely
    images, all_points = load_paired_images_and_points(folder)

    if len(images) < 2:
        raise RuntimeError(f"Not enough valid image+landmark pairs to average. valid_pairs={len(images)}")

    # Eye corners in output
    eyecorner_dst = [
        (int(0.3 * w), int(h / 3)),
        (int(0.7 * w), int(h / 3))
    ]

    images_norm = []
    points_norm = []

    boundary_pts = np.array([
        (0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
        (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)
    ], dtype=np.float32)

    points_avg = np.array([(0, 0)] * (68 + len(boundary_pts)), np.float32())
    num_images = len(images)

    for i in range(num_images):
        pts = all_points[i]  # list length 68
        eyecorner_src = [pts[36], pts[45]]

        tform = similarity_transform(eyecorner_src, eyecorner_dst)

        img_warp = cv2.warpAffine(
            images[i], tform, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        pts2 = np.reshape(np.array(pts, dtype=np.float32), (68, 1, 2))
        pts_warp = cv2.transform(pts2, tform)
        pts_warp = np.float32(np.reshape(pts_warp, (68, 2)))

        pts_warp = np.append(pts_warp, boundary_pts, axis=0)

        points_avg = points_avg + pts_warp / num_images
        points_norm.append(pts_warp)
        images_norm.append(img_warp)

    rect = (0, 0, w, h)
    tri = calculate_triangles(rect, np.array(points_avg, dtype=np.float32))
    if len(tri) == 0:
        raise RuntimeError("No Delaunay triangles found. Check landmarks / output size.")

    output = np.zeros((h, w, 3), np.float32())

    for i in range(num_images):
        img_acc = np.zeros((h, w, 3), np.float32())

        for j in range(len(tri)):
            t_in = []
            t_out = []
            for k in range(3):
                p_in = points_norm[i][tri[j][k]]
                p_out = points_avg[tri[j][k]]

                p_in = constrain_point(p_in, w, h)
                p_out = constrain_point(p_out, w, h)

                t_in.append(p_in)
                t_out.append(p_out)

            warp_triangle(images_norm[i], img_acc, t_in, t_out)

        output = output + img_acc

    output = output / num_images

    out_file = os.path.join(folder, "average_face.jpg")
    out_img = np.clip(255.0 * output, 0, 255).astype(np.uint8)
    imwrite_unicode(out_file, out_img)
    print(f"[OK] Saved average face: {out_file}")


def load_paired_images_and_points(folder: str):
    """
    Pair by landmark txt -> corresponding image file (txt without trailing .txt).
    Skips unreadable images or invalid landmark files.
    Uses Unicode-safe image read on Windows.
    """
    images = []
    points_list = []

    txt_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".txt")])

    skipped = 0
    for txt_name in txt_files:
        txt_path = os.path.join(folder, txt_name)
        img_name = txt_name[:-4]  # strip ".txt"
        img_path = os.path.join(folder, img_name)

        if not img_name.lower().endswith(IMG_EXTS):
            skipped += 1
            continue

        # read landmarks
        pts = []
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    x, y = parts
                    pts.append((float(x), float(y)))
        except Exception:
            skipped += 1
            continue

        if len(pts) != 68:
            skipped += 1
            continue

        # unicode-safe read (force 3-channel)
        img = imread_unicode_color(img_path)
        if img is None:
            skipped += 1
            continue

        img = np.float32(img) / 255.0  # float32 0..1, 3-ch
        images.append(img)
        points_list.append(pts)

    print(f"[INFO] valid_pairs={len(images)}, skipped_pairs={skipped}")
    return images, points_list


def similarity_transform(in_points, out_points):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(np.array(in_points, dtype=np.float32)).tolist()
    out_pts = np.copy(np.array(out_points, dtype=np.float32)).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]
    in_pts.append([xin, yin])

    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]
    out_pts.append([xout, yout])

    tform = cv2.estimateAffinePartial2D(np.array([in_pts], dtype=np.float32),
                                        np.array([out_pts], dtype=np.float32))
    return tform[0]


def rect_contains(rect, point):
    if point[0] < rect[0] or point[1] < rect[1] or point[0] > rect[2] or point[1] > rect[3]:
        return False
    return True


def calculate_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []

    for t in triangle_list:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if rect_contains(rect, pt[0]) and rect_contains(rect, pt[1]) and rect_contains(rect, pt[2]):
            ind = []
            for j in range(3):
                for k in range(len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
                        break
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))
    return delaunay_tri


def constrain_point(p, w, h):
    return (min(max(float(p[0]), 0.0), float(w - 1)), min(max(float(p[1]), 0.0), float(h - 1)))


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect, t2_rect, t2_rect_int = [], [], []
    for i in range(3):
        t1_rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
        t2_rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))
        t2_rect_int.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    # enforce 3-channel
    if img2_rect.ndim == 2:
        img2_rect = cv2.cvtColor(img2_rect, cv2.COLOR_GRAY2BGR)

    img2_rect = img2_rect * mask

    roi = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    roi = roi * (1.0 - mask) + img2_rect
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = roi


if __name__ == "__main__":
    main()
