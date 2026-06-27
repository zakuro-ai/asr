import argparse
import os

from gnutools import fs
from sklearn.model_selection import train_test_split

from asr_deepspeech import cfg
from asr_deepspeech.etl import JSUTDataset, LibriSpeechDataset

DATASETS = {
    "jsut": JSUTDataset,
    "librispeech": LibriSpeechDataset,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bronze-layer manifests")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="jsut",
        help="Which dataset to process (default: jsut)",
    )
    parser.add_argument("--landing", default=None, help="Override landing path from config")
    parser.add_argument("--bronze", default=None, help="Override bronze path from config")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from its public source into landing before processing",
    )
    parser.add_argument(
        "--subset",
        default="dev-clean-2",
        help="Subset to download (librispeech), e.g. dev-clean-2, dev-clean, test-clean",
    )
    parser.add_argument("--url", default=None, help="Override the download URL")
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    landing = args.landing or cfg.landing
    bronze = args.bronze or cfg.bronze

    dataset = DATASETS[args.dataset](cfg.fq)
    if args.download:
        print(f"Downloading {args.dataset} ({args.subset}) into {landing} ...")
        dataset.download(landing, subset=args.subset, url=args.url)
    dataset = dataset.run(landing, bronze)

    dataset_df = dataset.filter_duration(cfg.min_duration, cfg.max_duration)
    print(f"Filtered {len(dataset_df)} samples")
    print(dataset_df.head())

    os.makedirs(fs.parent(cfg.label_path), exist_ok=True)
    dataset.export_labels(cfg.label_path)
    dtrain, dtest = train_test_split(dataset_df, test_size=args.test_size)
    dtrain.to_csv(cfg.loaders.train_manifest, index=False)
    dtest.to_csv(cfg.loaders.val_manifest, index=False)
    print(f"Train: {len(dtrain)}  Test: {len(dtest)}")
