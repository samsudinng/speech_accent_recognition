# Speech Accent Classification with Image Classifier
Accent classification from spectrogram images with image classifier. For details on folder structure of the
repository and format of the configuration file, refer to [PyTorch Template Project readme file.](https://github.com/samsudinng/pytorch-template/blob/master/README.md)

## USAGE

#### 0. Install `conda` environment/package manager
This project uses `conda` to manage environment replication. Download and install from [here](https://docs.conda.io/en/latest/). 

#### 1. Setup virtual environment and install dependencies

```
conda create -n your_env_name python=3.7
source activate your_env_name
source setup_accent_env.sh your_env_name
```

#### 2. Source `wav_to_png.sh` to extract features

```
cd features_extraction
source wav_to_png.sh
```

This script performs two tasks:

- `wav_to_features.py`: convert .wav utterances (train, dev, test) into log-spectrogram features (\*.pkl.gz)
- `features_to_png.py`: read the features, segment and convert into .png images for model training and testing. The images are organized into folder corresponding to the label (according to `torchvision.datasets.ImageFolder` specification)

In the script, set the paths accordingly (absolute path or relative to the folder `features_extraction/`):

|Variable|Remarks|Default|
|:---|:---|:---|
|`TRAINWAVPATH`|path to .wav of train and dev set (see below for directory structure)|`audio/`|
|`TESTWAVPATH`|path to .wav of test |`testaudio/`|
|`FPATH`|path to save the feature files|`features/`|
|`TRAINIMGPATH`|path to save the trainset images (.png) for model training|`train_img/`|
|`DEVIMGPATH`|path to save the devset images (.png) for model validation|`dev_img/`|
|`TESTIMGPATH`|path to save the test images (.png) for model testing|`test_img/`|

```diff
- Note: `TRAINIMGPATH`, `DEVIMGPATH` and `TESTIMGPATH` must not be the same directory -
```

#### 3. Training/Validation

To train the model:

`python train.py -c path/to/config_conf-filename.json --trndir path/to/train_img/ --devdir path/to/dev_img/ --logdir path/to/save/logfiles/`

To resume training from specific epoch (eg.epoch number 6):

`python train.py -r path/to/checkpoint-epoch6.pth`

The required arguments are:

|Arguments|Remarks|Default|
|:---|:---|:--- |
|`--trndir`, `--devdir`|path to the trainset and devset images (`TRAINIMGPATH` and `DEVIMGPATH` in previous step)| `features_extraction/train_img/`<br/>`features_extraction/dev_img/`|
|`--logdir`|path to save logfiles, checkpoints and best model<br/><br/>In this directory, two subdirectories will be created:<br/>(1) `log/conf-filename/timestamp/` containing tensorboard log files and metric printout (`info.log`);<br/>(2) `models/conf-filename/timestamp/` containing checkpoints (`checkpoint-epoch<n>.pth`) and best model (`model_best.pth`) and a copy of the config file (`config.json`)| `saved/`|

```diff
- Config files to replicate various experiments are provided in the directory `config_files/` -
```

#### 4. Test

`python test.py -r path/to/model_best.pth ---tstdir path/to/testset_images/ `

By default, the .json config points to `features_extraction/test_img/` directory for test images. Results can be read from the file `info.log`

#### 5. Tensorboard visualization

By default, Tensorboard logging is enabled in the config files. Results can be monitored in Tensorboard with option `--logdir saved/`, or read from the file `saved/log/conf-filename/timestamp/info.log`. 

## OVERVIEW

#### DATASET
| | |
|:-|:-|
|__Source__ |Accented English Speech Recognition Challenge 2020  (Interspeech 2020)<br/>https://www.datatang.ai/INTERSPEECH2020 |
|__Audio__ | .wav (16 kHz, mono) |
|__Labels__|8 accented English utterances<br/>(Russia, Korea, US, Portugal, Japan, India, UK, China) |
|__Speakers__ |40 - 110 speakers per accent|
|__# Utterances__| Train set: 124k<br/>Dev set  : 12k<br/>Test set : 14.5k|


#### FEATURES

__`logspec200`__: Spectrogram of utterances, standardized (z-score based on train set), normalized (range: {0 ... 1}), quantized (8-bit, uint8)


__MODELS__

The available and tested models are listed below. 

|Model|Description|Source|
|:---|:---|:---|
|`AlexNetGAP`|AlexNet features layer + global average pooling classifier|torchvision.models.alexnet|
|`VGG16GAP`|VGG16 features layer + global average pooling classifier|torchvision.models.vgg16|
|`VGG16BnGAP`|VGG16+Batchnorm + global average pooling classifier|torchvision.models.vgg16bn|
|`Resnet34`| |torchvision.models.resnet34|
|`Resnet50`| |torchvision.models.resnet50|
|`Resnet101`| |torchvision.models.resnet101|
|`Resnet152`| |torchvision.models.resnet152|

__PERFORMANCE METRIC__

- Training  :  __`accuracy`__ (overall accuracy, segmental level)
- Dev/Test  :  __`utt_accuracy`__ (overal accuracy on utterance level, i.e. posterior probabilities from all segments of the utterance)






