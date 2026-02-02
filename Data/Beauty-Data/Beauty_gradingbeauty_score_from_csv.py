#!/usr/bin/env python3
"""Append SCUT-FBP5500 beauty scores to a CSV of image paths.

Example:
  python beauty_score_from_csv.py \
    --input-csv professors.csv \
    --image-column image_path \
    --output-csv professors_with_scores.csv \
    --model-arch alexnet \
    --weights ./models/alexnet.pth
"""

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

import scut_fbp5500_model


MODEL_CHOICES = {
    "alexnet": lambda: scut_fbp5500_model.AlexNet(num_classes=1),
    "resnet18": lambda: scut_fbp5500_model.ResNet(
        block=scut_fbp5500_model.BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1,
    ),
}


def load_model(weights_path: str, model_arch: str, device: torch.device) -> torch.nn.Module:
    if model_arch not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    model = MODEL_CHOICES[model_arch]().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model_state = model.state_dict()
    if "state_dict" in checkpoint:
        pretrained = checkpoint["state_dict"]
    else:
        pretrained = checkpoint
    filtered = {k: v for k, v in pretrained.items() if k in model_state}
    model_state.update(filtered)
    model.load_state_dict(model_state)
    model.eval()
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row.")
        return rows, reader.fieldnames


def write_csv_rows(path: str, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(image_path: str, base_dir: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.normpath(os.path.join(base_dir, image_path))


def predict_scores(
    rows: List[Dict[str, str]],
    image_column: str,
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose,
    base_dir: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    errors: List[str] = []
    scored_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        if image_column not in row:
            raise KeyError(f"Image column '{image_column}' not found in CSV.")

        image_path = row[image_column]
        resolved_path = resolve_image_path(image_path, base_dir)
        score_value: float
        try:
            with Image.open(resolved_path) as img:
                img_rgb = img.convert("RGB")
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor).squeeze(1).cpu().item()
            score_value = float(output)
        except Exception as exc:  # pylint: disable=broad-except
            score_value = float("nan")
            errors.append(f"Row {index}: {image_path} -> {exc}")

        new_row = dict(row)
        new_row["beauty_score"] = "" if math.isnan(score_value) else f"{score_value:.4f}"
        scored_rows.append(new_row)

    return scored_rows, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append SCUT-FBP5500 beauty scores to a CSV of image paths."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path.")
    parser.add_argument(
        "--output-csv",
        help="Output CSV path (defaults to <input>_with_scores.csv).",
    )
    parser.add_argument(
        "--image-column",
        default="image_path",
        help="Column containing image paths (default: image_path).",
    )
    parser.add_argument(
        "--model-arch",
        choices=sorted(MODEL_CHOICES.keys()),
        default="alexnet",
        help="Model architecture to load (default: alexnet).",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to pretrained SCUT-FBP5500 weights (.pth).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if not output_csv:
        base, ext = os.path.splitext(args.input_csv)
        output_csv = f"{base}_with_scores{ext or '.csv'}"

    device = torch.device(args.device)
    model = load_model(args.weights, args.model_arch, device)
    transform = build_transform()

    rows, fieldnames = read_csv_rows(args.input_csv)
    base_dir = os.path.dirname(os.path.abspath(args.input_csv))
    scored_rows, errors = predict_scores(
        rows,
        args.image_column,
        model,
        device,
        transform,
        base_dir,
    )

    if "beauty_score" not in fieldnames:
        fieldnames = fieldnames + ["beauty_score"]

    write_csv_rows(output_csv, scored_rows, fieldnames)

    if errors:
        error_log = os.path.splitext(output_csv)[0] + "_errors.txt"
        with open(error_log, "w", encoding="utf-8") as handle:
            handle.write("\n".join(errors))
        print(f"Completed with {len(errors)} errors. See {error_log}.")
    else:
        print(f"Completed successfully. Output written to {output_csv}.")


if __name__ == "__main__":
    main()
