#!/usr/bin/env python3
"""
Обучение YOLOv11 на датасете watermark'ов AI-генераторов.

Использование:
  python train_watermark.py                                    # дефолтные параметры
  python train_watermark.py --data datasets/watermarks/data.yaml
  python train_watermark.py --model yolo11s.pt --epochs 150   # побольше и помощнее
  python train_watermark.py --device cuda:0                   # если есть GPU

После обучения лучшие веса автоматически копируются в models/watermark_yolo.pt.
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO watermark detector")
    p.add_argument("--data",   default="datasets/watermarks/data.yaml")
    p.add_argument("--model",  default="yolo11n.pt",
                   help="Базовая модель: yolo11n.pt (быстро) / yolo11s.pt / yolo11m.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", default="cpu", help="cpu / cuda:0 / mps")
    p.add_argument("--output", default="models/watermark_yolo.pt",
                   help="Куда сохранить финальные веса")
    p.add_argument("--tiny",   action="store_true",
                   help="Режим для маленьких датасетов (< 50 фото/класс)")
    return p.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Установи ultralytics: pip install ultralytics")
        sys.exit(1)

    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Не найден data.yaml: {data_path}")
        print("Скачай датасет: python download_dataset.py --help")
        sys.exit(1)

    if args.tiny:
        # Режим для маленьких датасетов (10–50 фото/класс):
        # - маленький batch чтобы не вылететь по памяти
        # - больше эпох + mosaic не отключаем до конца
        # - ниже LR чтобы не переобучиться сразу
        epochs  = args.epochs if args.epochs != 100 else 300
        batch   = 4
        patience = 60
        lr0     = 0.001
        close_mosaic = 0   # mosaic работает до последней эпохи
        warmup  = 5
    else:
        epochs  = args.epochs
        batch   = args.batch
        patience = 30
        lr0     = 0.01
        close_mosaic = 10
        warmup  = 3

    mode = "TINY (мало данных)" if args.tiny else "стандартный"
    print(f"Модель:  {args.model}  [{mode}]")
    print(f"Данные:  {data_path}")
    print(f"Эпохи:   {epochs}  Batch: {batch}  imgsz: {args.imgsz}")
    print(f"Девайс:  {args.device}\n")

    if args.tiny:
        print("⚠ Tiny-режим: 10 фото/класс даст низкую точность.")
        print("  Характерные watermarks (DALL-E 2 квадраты, текст Midjourney) скорее всего сработают.")
        print("  Остальные — как повезёт. Добавь больше фото если accuracy не устроит.\n")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/watermark",
        name="train",
        exist_ok=True,

        # Watermark'и обычно в углу → вертикальный flip ломает паттерн
        flipud=0.0,
        fliplr=0.5,

        # Небольшие геометрические искажения — логотипы слегка вращаются/масштабируются
        degrees=5.0,
        translate=0.1,
        scale=0.5,

        # Цветовые аугментации — компенсируют разные условия скриншотов
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        patience=patience,
        close_mosaic=close_mosaic,
        lr0=lr0,
        lrf=0.01,
        warmup_epochs=warmup,
    )

    # Копируем best.pt → models/watermark_yolo.pt
    best = Path("runs/watermark/train/weights/best.pt")
    if best.exists():
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, dest)
        print(f"\n✓ Веса сохранены: {dest}")
        print(f"  Сервер автоматически подхватит их при следующем запуске.")
    else:
        print(f"\n✗ best.pt не найден в {best.parent} — проверь логи обучения")
        sys.exit(1)


if __name__ == "__main__":
    main()
