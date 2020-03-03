![alt text](https://github.com/JeanMaximilienCadic/ASRDeepSpeech/blob/master/img/Training-Deep-Speech.png)
# ASRDeepspeech (English/Japanese)

This repository offers a clean code version of the original repository from SeanNaren with classes and modular
components (eg trainers, models, loggers...).

## Improvements
<li> Clean verbose during training 

```
================ VARS ===================
manifest: clean
distributed: True
train_manifest: __data__/manifests/train_clean.json
val_manifest: __data__/manifests/val_clean.json
model_path: /data/ASRModels/deepspeech_jp_500_clean.pth
continue_from: None
output_file: /data/ASRModels/deepspeech_jp_500_clean.txt
main_proc: True
rank: 0
gpu_rank: 0
world_size: 2
==========================================
```
<li> Progress bar

```
...
clean - 0:01:07 >> 809/1000 (1) | Loss 96.9079 | Lr 0.30e-3 | WER/CER 28.57/26.91 - (28.57/[26.91]): 100%|██████████| 18/18 [01:05<00:00,  3.62s/it]
clean - 0:01:05 >> 810/1000 (2) | Loss 96.6589 | Lr 0.29e-3 | WER/CER 28.06/26.41 - (28.06/[26.41]):  61%|██████    | 11/18 [00:41<00:25,  3.65s/it]
...
```

<li> Separated text file to check wer/cer (best/last/worst result)

```
================= WER / CER =================
----- BEST -----
Ref:思 い 出 し た よ
Hyp:   した
WER:100.0  - CER:66.66666666666666
----- LAST -----
Ref:そ れ を 開 き
Hyp: た
WER:100.0  - CER:100.0
----- WORST -----
Ref:へ 一 そ っ か 今 あ れ だ よ ね だ っ て さ み 一 ん な や っ て る も ん ね
Hyp:     た
WER:100.0  - CER:100.0
==============================================
```



## Installation

### From Source

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu, with Pytorch 1.0.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Compile and install the dependencies
```
bash ./setup_dependencies.sh
```

Install the python requirements
```
pip install -r requirements.txt
```

## Datasets

Currently supports JSUT. Please contact me if you want to download the preprocessed files and jp_labels.json.

#### Custom Dataset

To create a custom dataset you must create json files containing the necessary information about the dataset.
> \_\_data\_\_/manifests/train_jsut.json
```
{"audio_filepath": "/path/to/audio_train1.wav", "duration": seconds1, "text": "train string content 1"}
{"audio_filepath": "/path/to/audio_train2.wav", "duration": seconds2, "text": "train string content 2"}
{"audio_filepath": "/path/to/audio_train3.wav", "duration": seconds3, "text": "train string content 3"}
...
```
> \_\_data\_\_/manifests/test_jsut.json
```
{"audio_filepath": "/path/to/audio_test1.wav", "duration": seconds1, "text": "test string content 1"}
{"audio_filepath": "/path/to/audio_test2.wav", "duration": seconds2, "text": "test string content 2"}
{"audio_filepath": "/path/to/audio_test3.wav", "duration": seconds3, "text": "test string content 3"}
...
```

## Training a Model

To train on a single gpu
```
python train.py --manifest [manifest_id] --labels [path_to_labels_json]
```
To scale to multi-gpu training
```
python -m multiproc train.py --manifest [manifest_id] --labels [path_to_labels_json]             
```

## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!

This is a fork from https://github.com/SeanNaren/deepspeech.pytorch. The code has been improved for the readability only.

For any question please contact me at j.cadic[at]protonmail.ch
