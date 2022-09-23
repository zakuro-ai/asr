import os
from sklearn.model_selection import train_test_split
from asr_deepspeech.etl import JSUTDataset
from asr_deepspeech import cfg
from gnutools import fs
from gnutools.remote import gdrive

if __name__ == "__main__":
    # Download the dataset if needed
    if cfg.gdrive_uri is not None:
        gdrive(cfg.gdrive_uri)

    # Landing / bronze
    dataset = JSUTDataset(cfg.fq)
    dataset = dataset.run(cfg.landing, cfg.bronze)

    # Silver (cache)
    dataset_df = dataset.filter_duration(cfg.min_duration, cfg.max_duration)
    print(dataset_df.head())

    # Gold
    os.makedirs(fs.parent(cfg.label_path), exist_ok=True)
    dataset.export_labels(cfg.label_path)
    dtrain, dtest = train_test_split(dataset_df, test_size=0.1)
    dtrain.to_csv(cfg.loaders.train_manifest, index=False)
    dtest.to_csv(cfg.loaders.val_manifest, index=False)
