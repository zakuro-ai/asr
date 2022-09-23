<h1 align="center">
  <br>
  <img src="https://drive.google.com/uc?id=17SeD6ijR7DV_EnZGJqavHxVbNHs8n4EQ">
  <br>
    ASRDeepspeech x Sakura-ML 
    (English/Japanese)
  <br>
</h1>

<p align="center">
  <a href="#modules">Modules</a> •
  <a href="#code-structure">Code structure</a> •
  <a href="#installing-the-application">Installing the application</a> •
  <a href="#makefile-commands">Makefile commands</a> •
  <a href="#environments">Environments</a> •
  <a href="#dataset">Dataset</a>•
  <a href="#running-the-application">Running the application</a>•
  <a href="#notes">Notes</a>•
</p>


This repository offers a clean code version of the original repository from SeanNaren with classes and modular
components (eg trainers, models, loggers...).

I have added a configuration file to manage the parameters set in the model. You will also find a pretrained model in japanese performing a `CER = 34` on JSUT test set .

# Modules

At a granular level, ASRDeepSpeech is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **asr_deepspeech** | Speech Recognition package |
| **asr_deepspeech.data** | Data related module |
| **asr_deepspeech.data.dataset** | Build the dataset |
| **asr_deepspeech.data.loaders** | Load the dataset |
| **asr_deepspeech.data.parsers** | Parse the dataset |
| **asr_deepspeech.data.samplers** | Sample the dataset |
| **asr_deepspeech.decoders** | Decode the generated text |
| **asr_deepspeech.loggers** | Loggers |
| **asr_deepspeech.modules** | Components of the network |
| **asr_deepspeech.parsers** | Arguments parser |
| **asr_deepspeech.tests** | Test units |
| **asr_deepspeech.trainers** | Trainers |


# Code structure
```python
from setuptools import setup
from asr_deepspeech import __version__

setup(
    name="asr_deepspeech",
    version=__version__,
    short_description="ASRDeepspeech (English / Japanese)",
    long_description="".join(open("README.md", "r").readlines()),
    long_description_content_type="text/markdown",
    url="https://github.com/zakuro-ai/asr",
    license="MIT Licence",
    author="CADIC Jean-Maximilien",
    python_requires=">=3.8",
    packages=[
        "asr_deepspeech",
        "asr_deepspeech.audio",
        "asr_deepspeech.data",
        "asr_deepspeech.data.dataset",
        "asr_deepspeech.data.loaders",
        "asr_deepspeech.data.manifests",
        "asr_deepspeech.data.parsers",
        "asr_deepspeech.data.samplers",
        "asr_deepspeech.decoders",
        "asr_deepspeech.etl",
        "asr_deepspeech.loggers",
        "asr_deepspeech.models",
        "asr_deepspeech.modules",
        "asr_deepspeech.parsers",
        "asr_deepspeech.tests",
        "asr_deepspeech.trainers",
    ],
    include_package_data=True,
    package_data={"": ["*.yml"]},
    install_requires=[r.rsplit()[0] for r in open("requirements.txt")],
    author_email="git@zakuro.ai",
    description="ASRDeepspeech (English / Japanese)",
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)

```


# Installing the application
To clone and run this application, you'll need the following installed on your computer:
- [Git](https://git-scm.com)
- Docker Desktop
   - [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
   - [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux-install/)
- [Python](https://www.python.org/downloads/)

Install bpd:
```bash
# Clone this repository and install the code
git clone https://github.com/zakuro-ai/asr

# Go into the repository
cd asr
```


# Makefile commands
Exhaustive list of make commands:
```
pull                # Download the docker image
sandbox             # Launch the sandox image 
install_wheels      # Install the wheel
tests               # Test the code
```
# Environments
We are providing a support for local or docker setup. However we recommend to use docker to avoid any difficulty to run
 the code. 
If you decide to run the code locally you will need python3.6 with cuda>=10.1.
Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu, with Pytorch 1.0.
Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

## Docker

> **Note**
> 
> Running this application by using Docker is recommended.

To build and run the docker image
```
make pull
make sandbox
```

## PythonEnv

> **Warning**
> 
> Running this application by using PythonEnv is possible but *not* recommended.
```
make install_wheels
```

## Test
```
make tests
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

# Datasets

By default we process the JSUT dataset. See the `config` section to know how to process a custom dataset.
```python
from gnutools.remote import gdrive
from asr_deepspech import  cfg

# This will download the JSUT dataset in your /tmp
gdrive(cfg.gdrive_uri)
```
## ETL

```
python -m asr_deepspeech.et
```

# Running the application

## Training a Model

To train on a single gpu
```bash
sakura -m asr_deepspeech.trainers
```

## Pretrained model
```bash
python -m asr_deepspeech
```


# Notes
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
