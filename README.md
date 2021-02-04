![alt text](img/Training-Deep-Speech.png)
# ASRDeepspeech (English/Japanese)

This repository offers a clean code version of the original repository from SeanNaren with classes and modular
components (eg trainers, models, loggers...).

## Overview
## ASRDeepspeech modules

At a granular level, synskit is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **asr_deepspeech** | |
| **asr_deepspeech.data** | |
| **asr_deepspeech.data.dataset** | |
| **asr_deepspeech.data.loaders** | |
| **asr_deepspeech.data.parsers** | |
| **asr_deepspeech.data.samplers** | |
| **asr_deepspeech.decoders** | |
| **asr_deepspeech.loggers** | |
| **asr_deepspeech.models** | |
| **asr_deepspeech.modules** | |
| **asr_deepspeech.parsers** | |
| **asr_deepspeech.test** | |
| **asr_deepspeech.trainers** | |

## Installation
We are providing a support for local or docker setup. However we recommend to use docker to avoid any difficulty to run
 the code. 
 
 If you decide to run the code locally you will need python3.6 with cuda>=10.1.
#### Docker
To build the image with docker
```
docker build . -t jcadic/deepspeech
```


#### Local 
```
sh setup.sh
```
##Test the setup
#### Docker
```
docker run --rm --gpus all -it jcadic/deepspeech
```
#### Local
```
python -m asr_deepspeech.test
```

You should be able to get an output like
```
=1= TEST PASSED : asr_deepspeech
=1= TEST PASSED : asr_deepspeech.data
=1= TEST PASSED : asr_deepspeech.data.dataset
=1= TEST PASSED : asr_deepspeech.data.loaders
=1= TEST PASSED : asr_deepspeech.data.parsers
=1= TEST PASSED : asr_deepspeech.data.samplers
=1= TEST PASSED : asr_deepspeech.decoders
=1= TEST PASSED : asr_deepspeech.loggers
=1= TEST PASSED : asr_deepspeech.models
=1= TEST PASSED : asr_deepspeech.modules
=1= TEST PASSED : asr_deepspeech.parsers
=1= TEST PASSED : asr_deepspeech.test
=1= TEST PASSED : asr_deepspeech.trainers

```
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
clean - 0:00:46 >> 2/1000 (1) | Loss 95.1626 | Lr 0.30e-3 | WER/CER 98.06/95.16 - (98.06/[95.16]): 100%|██████████████████████| 18/18 [00:46<00:00,  2.59s/it]
clean - 0:00:47 >> 3/1000 (1) | Loss 96.3579 | Lr 0.29e-3 | WER/CER 97.55/97.55 - (98.06/[95.16]): 100%|██████████████████████| 18/18 [00:47<00:00,  2.61s/it]
clean - 0:00:47 >> 4/1000 (1) | Loss 97.5705 | Lr 0.29e-3 | WER/CER 100.00/100.00 - (98.06/[95.16]): 100%|████████████████████| 18/18 [00:47<00:00,  2.66s/it]
clean - 0:00:48 >> 5/1000 (1) | Loss 97.8628 | Lr 0.29e-3 | WER/CER 98.74/98.74 - (98.06/[95.16]): 100%|██████████████████████| 18/18 [00:50<00:00,  2.78s/it]
clean - 0:00:50 >> 6/1000 (5) | Loss 97.0118 | Lr 0.29e-3 | WER/CER 96.26/93.61 - (96.26/[93.61]): 100%|██████████████████████| 18/18 [00:49<00:00,  2.76s/it]
clean - 0:00:50 >> 7/1000 (5) | Loss 97.2341 | Lr 0.28e-3 | WER/CER 98.35/98.35 - (96.26/[93.61]):  17%|███▊                   | 3/18 [00:10<00:55,  3.72s/it]
...
```

<li> Separated text file to check wer/cer with histogram on CER values (best/last/worst result)

```
================= 43.52/43.55 =================
----- BEST -----
Ref:や さ し い ほ し は こ た え ま し た
Hyp:や さ し い ほ し は こ た え ま し た
WER:0.0  - CER:0.0
----- LAST -----
Ref:そ れ を 開 き
Hyp:そ れ け
WER:60.0  - CER:60.0
----- WORST -----
Ref:サ ル ト サ ム ラ イ
Hyp:死 る と さ む ら い
WER:100.0  - CER:100.0
CER histogram
|###############################################################################
|█████████████████████████████████████                              144  0-10
|███████████████████████████████████████████████████████            212  10-20
|█████████████████████████████████████████████████████████████████  249  20-30
|█████████████████████████████████████████████████████████          222  30-40
|███████████████████████████████████████████                        168  40-50
|████████████████████████████████████████████████████               203  50-60
|███████████████████████████████████████████                        167  60-70
|████████████████████████████████                                   126  70-80
|████████████████████████████                                       110  80-90
|██████                                                              26  90-100
|████████████████████                                                78  100-110
=============================================

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
