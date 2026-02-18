import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from torchvision import datasets, transforms

from resnet101.src.data.utils_data import get_or_make_stats


def ingest_oxford_pets(
    data_dir: str,
    stats_path: str,
    report_path: str,
    img_size: int = 224,
    robust: bool = False,
    num_workers: int = 2,
    force_recompute_stats: bool = False,
):
    data_dir_path = Path(data_dir)
    stats_path_obj = Path(stats_path)
    report_path_obj = Path(report_path)

    data_dir_path.mkdir(parents=True, exist_ok=True)
    stats_path_obj.parent.mkdir(parents=True, exist_ok=True)
    report_path_obj.parent.mkdir(parents=True, exist_ok=True)

    tf_tmp = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )

    trainval_ds = datasets.OxfordIIITPet(
        root=str(data_dir_path),
        split="trainval",
        target_types="category",
        download=True,
        transform=tf_tmp,
    )
    test_ds = datasets.OxfordIIITPet(
        root=str(data_dir_path),
        split="test",
        target_types="category",
        download=True,
        transform=tf_tmp,
    )

    cache_path = None if force_recompute_stats else str(stats_path_obj)
    loc, scale = get_or_make_stats(
        trainval_ds,
        cache_path=cache_path,
        robust=robust,
        tmp_bs=64,
        num_workers=num_workers,
    )

    if force_recompute_stats:
        stats_payload = {"robust": robust, "loc": loc, "scale": scale, "img_size": img_size}
        stats_path_obj.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "OxfordIIITPet",
        "data_dir": str(data_dir_path),
        "n_trainval": len(trainval_ds),
        "n_test": len(test_ds),
        "stats_path": str(stats_path_obj),
        "robust": robust,
        "img_size": img_size,
        "loc": loc,
        "scale": scale,
    }
    report_path_obj.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Dataset descargado/preparado en: {data_dir_path}")
    print(f"[OK] Stats disponibles en: {stats_path_obj}")
    print(f"[OK] Reporte de ingesta: {report_path_obj}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingesta de datos Oxford Pets y generacion de stats de normalizacion."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--stats-path", type=str, default="data/pet_stats.json")
    parser.add_argument("--report-path", type=str, default="monitoring/ingestion_report.json")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-recompute-stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ingest_oxford_pets(
        data_dir=args.data_dir,
        stats_path=args.stats_path,
        report_path=args.report_path,
        img_size=args.img_size,
        robust=args.robust,
        num_workers=args.num_workers,
        force_recompute_stats=args.force_recompute_stats,
    )


if __name__ == "__main__":
    main()
