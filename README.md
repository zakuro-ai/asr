![alt text](img/Training-Deep-Speech.png)
# ASRDeepspeech (English/Japanese)

This repository offers a clean code version of the original repository from SeanNaren with classes and modular
components (eg trainers, models, loggers...).

I have added a configuration file to manage the parameters set in the model. You will also find a pretrained model in japanese performing a `CER = 34` on JSUT test set .

## Overview
## ASRDeepspeech modules

At a granular level, ASRDeepSpeech is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **asr_deepspeech** | Speech Recognition package|
| **asr_deepspeech.data** | Data related module|
| **asr_deepspeech.data.dataset** | Build the dataset|
| **asr_deepspeech.data.loaders** | Load the dataet|
| **asr_deepspeech.data.parsers** | Parse the dataset|
| **asr_deepspeech.data.samplers** | Sample the dataset|
| **asr_deepspeech.decoders** | Decode the generated text |
| **asr_deepspeech.loggers** | Loggers |
| **asr_deepspeech.modules** | Components of the network|
| **asr_deepspeech.parsers** | Arguments parser|
| **asr_deepspeech.test** | Test units|
| **asr_deepspeech.trainers** | Trainers |

## Installation
We are providing a support for local or docker setup. However we recommend to use docker to avoid any difficulty to run
 the code. 
If you decide to run the code locally you will need python3.6 with cuda>=10.1.
Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu, with Pytorch 1.0.
Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

#### Docker
To build the image with docker, download the pretrained model in japanese and check the `WER/CER` on JSUT test set.
```Dockerfile
docker rmi -f jmcadic/deepspeech
docker build . -t jmcadic/deepspeech
docker run \
  --rm \
  --gpus "device=1" \
  -it \
  --shm-size=70g \
  -v $(pwd):/workspace \
  -v /srv/sync/:/srv/sync \
  -v $HOME/.zakuro:/root/.zakuro \
   jmcadic/deepspeech  python -m asr_deepspeech
```


#### Local 
```bash
sh setup.sh
python -m asr_deepspeech.test
```

You should be able to get an output like
```python
=1= TEST PASSED : asr_deepspeech
=1= TEST PASSED : asr_deepspeech.data
=1= TEST PASSED : asr_deepspeech.data.dataset
=1= TEST PASSED : asr_deepspeech.data.loaders
=1= TEST PASSED : asr_deepspeech.data.parsers
=1= TEST PASSED : asr_deepspeech.data.samplers
=1= TEST PASSED : asr_deepspeech.decoders
=1= TEST PASSED : asr_deepspeech.loggers
=1= TEST PASSED : asr_deepspeech.modules
=1= TEST PASSED : asr_deepspeech.parsers
=1= TEST PASSED : asr_deepspeech.test
=1= TEST PASSED : asr_deepspeech.trainers
```

## Datasets

Currently supports JSUT. Please contact me if you want to download the preprocessed files and jp_labels.json.
```bash
wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
```
#### Custom Dataset

To create a custom dataset you must create json files containing the necessary information about the dataset. `__data__/manifests/{train/val}_jsut.json`
```json
{
    "UT-PARAPHRASE-sent002-phrase1": {
        "audio_filepath": "/mnt/.cdata/ASR/ja/raw/CLEAN/JSUT/jsut_ver1.1/utparaphrase512/wav/UT-PARAPHRASE-sent002-phrase1.wav",
        "duration": 2.44,
        "text": "専門には、疎いんだから。"
    },
    "UT-PARAPHRASE-sent002-phrase2": {
        "audio_filepath": "/mnt/.cdata/ASR/ja/raw/CLEAN/JSUT/jsut_ver1.1/utparaphrase512/wav/UT-PARAPHRASE-sent002-phrase2.wav",
        "duration": 2.82,
        "text": "専門には、詳しくないんだから。"
    },
    ...
}
```

## Training a Model

To train on a single gpu
```bash
python -m asr_deepspeech.trainers
```

## Pretrained model
This will load the `config.yml` containing the list of arguments for the inference and run a pretrained model.  
```bash
python -m asr_deepspeech
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
================= 100.00/34.49 =================
----- BEST -----
Ref:良ある人ならそんな風にに話しかけないだろう
Hyp:用ある人ならそんな風にに話しかけないだろう
WER:100.0  - CER:4.761904761904762
----- LAST -----
Ref:すみませんがオースチンさんは5日にはです
Hyp:すみませんがースンさんは一つかにはです
WER:100.0  - CER:25.0
----- WORST -----
Ref:小切には内がみられる
Hyp:コには内先金地つ作みが見られる
WER:100.0  - CER:90.0
CER histogram
|###############################################################################
|███████████                                                           6  0-10  
|███████████████████████████                                          15  10-20 
|███████████████████████████████████████████████████████████████████  36  20-30 
|█████████████████████████████████████████████████████████████████    35  30-40 
|██████████████████████████████████████████████████                   27  40-50 
|█████████████████████████████                                        16  50-60 
|█████████                                                             5  60-70 
|███████████                                                           6  70-80 
|                                                                      0  80-90 
|█                                                                     1  90-100
=============================================
```


## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!

This is a fork from https://github.com/SeanNaren/deepspeech.pytorch. The code has been improved for the readability only.

For any question please contact me at j.cadic[at]protonmail.ch
