#!/usr/bin/env python3
"""
Скачивание датасета watermark'ов с Roboflow Universe.

Шаги:
  1. Зайди на https://universe.roboflow.com
  2. Поищи: "AI watermark detection" или "watermark logo detection"
  3. Выбери датасет → Export → Format: YOLOv11 → нажми "Get Download Code"
  4. Скопируй workspace, project и version из кода который покажет Roboflow
  5. Получи API-ключ: https://app.roboflow.com/settings/api

Запуск:
  pip install roboflow
  python download_dataset.py \
    --api-key rf_XXXXXXXXXXXX \
    --workspace my-workspace \
    --project watermark-detection \
    --version 1

Или вручную: Export → Download ZIP → распакуй в datasets/watermarks/
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Roboflow dataset for YOLO training")
    p.add_argument("--api-key",   required=True,         help="Roboflow API key (rf_...)")
    p.add_argument("--workspace", required=True,         help="Workspace slug (из URL)")
    p.add_argument("--project",   required=True,         help="Project slug (из URL)")
    p.add_argument("--version",   type=int, default=1,   help="Версия датасета")
    p.add_argument("--output",    default="datasets/watermarks", help="Куда сохранить")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Сначала установи: pip install roboflow")
        sys.exit(1)

    print(f"Скачиваю {args.workspace}/{args.project} v{args.version}...")
    rf = Roboflow(api_key=args.api_key)
    dataset = (
        rf.workspace(args.workspace)
        .project(args.project)
        .version(args.version)
        .download("yolov11", location=args.output, overwrite=True)
    )

    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"\n✓ Датасет: {dataset.location}")
    print(f"  data.yaml: {data_yaml}")
    print(f"\nЗапускай обучение:")
    print(f"  python train_watermark.py --data {data_yaml}")


if __name__ == "__main__":
    main()
