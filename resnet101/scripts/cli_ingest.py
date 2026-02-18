import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CLI ResNet101 para ingesta de Oxford Pets.")
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
    from resnet101.src.data.ingest_data import ingest_oxford_pets

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
