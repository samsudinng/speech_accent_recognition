# Speech Accent Classification with Image Classifier
Accent classification from speech spectral images with image classifier

## 0. Overview

In this method, speech accent classification task  is formulated as image classification task.

__Dataset__

Accented English Speech Recognition Challenge 2020  (IS20)

__Input image__

- `logspec200': Spectrogram of utterances, converted into 8-bit RGB image (.png)
- `kaldi83' : Kaldi features (fbank+pitch, converted into 8-bit RGB image (.png)) *\*to be updated*

__Image classification model__

- AlexNetGAP

__Performance Metric__

- Training  :  `accuracy` (overall accuracy, segmental level)
- Dev/Test  :  `utt_accuracy` (overal accuracy on utterance level, i.e. posterior probabilities from all segments of the utterance)

__How to run/reproduce results?__

Follow these steps, details on following sections.

1. Extract features from .wav files
2. Convert features to .png images
3. Train/dev epochs
4. Evaluate on test set


## 1. Extract features from .wav

### 1.1 `logspec200` features (200-dim spectrogram)

__Notebook__: __Logspec_Features_from_Audio.jpynb__

__Input__: utterances .wav files, test\dev\train set listed in `metadata\*_utt2label`

__Output__: spectrograms in .pkl.gz files (train set splitted into 8 chunks due to large size), data type: uint8 (formatted as gray image pixel)

Run the notebook `__features_extraction\Logspec_Features_from_Audio.ipynb__`. The spectrograms are saved as .pkl.gz located in the folder `features`.

For detailed steps, check the notebook. The required folders are organized as follows: 

|Folder|Content|Action|
|:---|:---|:---|
|__./audio__|train and dev sets .wav files (see below for folder structure)|
|__./testaudio__|test set .wav files|
|__./metadata__ (provided in repo)|various metadata: `utt`, `label`, `sex`, `age`, utt to file paths for test set|
|__./features__| the extracted features (\*.pkl.gz)|

The folder structure for the train and dev dataset is as follows:
```
audio/  
   +---CHN/
   |     +----G0021/
   |             |------G00021S1053.wav
   |             |------   .
   |             |------   .
   +---IND
   +--- .
   +--- .
   +---US
``` 


#### 1.2.2 VGGVox magnitude-spectrogram features (512-dim)

Adapted from:
https://github.com/Derpimort/VGGVox-PyTorch

The author converted the Matlab original implementation to Python (and verified). The pre-trained weights was trained on VoxCeleb1 dataset for speaker identification/verification.

VGGVox features is magnitude spectrogram (512x300) with type `np.float32`. The frequency bins (512) is full mirrored spectrum (256+256) from 512-point FFT. Only the first 257 points are saved (spec\[:257,:\]) and the full features can be built by mirroring spec\[1:256,:\] and concatenating to the saved features.

__Notebook__: __VGGVox_Features_from_Audio.jpynb__

__Input__   : utterances .wav files, test set listed in `features_extraction\train\utt2label` and `features_extraction\dev\utt2label`

__Output__  : spectrograms in .pkl.gz files (splitted into 16 chunks due to large size), data type: float32 (raw spectrograms)

__Process__ :

   - dc removal and dithering
   - pre-emphasis filtering
   - convert to spectrogram
   - zscore standardization (per utterance)
   - save all quantized features as dictionary (```all_spec```)
      - key  : utterance name (eg. ```AESRC2020-AMERICAN-ACCENT-G00473-G00473S1028```)
      - value: numpy array (`np.float32`), shape = (F, T) where F = 257 and should be mirrored to 512 as VGGVox feature
   - write to .pkl.gz


## 2. Training/Validation

### 2.1 Features to image

#### 2.1.1 Using `logspec200` (spectrogram-based features)

For classification with __AlexNet__, the extracted features is converted into `.png` images and stored in folder `train_img\x\` and `dev_img\x\` where `x` is the label ranging from `0` to `7`. This is folloowing the folder structure to use `ImageFolder` dataset from `PyTorch`. 

To perform this step, run `python create_kaldi_img.py` or `python create_logspec_img.py` accordingly.

#### 2.1.2 Using Kaldi-based features (83-dim fbank + pitch)

To extract image from Kaldi feature matrix, simply update the variables in the script `create_kaldi_img.py` and run the script. The variables are:

|Variables|Meaning|
|---------|---------|
|`segment_size`|the dimension of non-overlapping kaldi feature image segment (time direction). Image size = (83 x segment_size) pixels|
|`train_path`, `dev_path`| path to the folder containing train/dev .ark and .scp files|
|`train_label_file`, `dev_label_file`| path to train/dev utt2accent file|
|`n_train_feats_files`, `n_train_feats_files`| number of train/dev .scp file segments|
|`train_dir`, `dev_dir`|directory to save the resulting train/dev images| 
|`old_train_path`, `old_dev_path`|the original path to .ark files as pointed in.scp, to be replaced with current train_path and dev_path|
 
The feature images will be save to the directory pointed by `train_dir` and `dev_dir`. In either directory, the images will be saved into subdirectory which 
corresponds to the label eg. if a training feature image has a label '0' it will be saved in `train_dir\0\` folder. This is following the image path organization 
for PyTorch ImageFolder dataset, which is used in this project.

### 2.2 Model Training and Validation

To train the model, try `python train.py -c config.json`. The training/validation cofiguration can be set in `config.json`. For details on folder structure of the
code and format of the configuration file, refer to [PyTorch Template Project readme file.](https://github.com/samsudinng/pytorch-template/blob/master/README.md)

Additional configuration has been added for mixup and test-time data augmentation implementation. The following can be enabled/disabled by setting some parameters
in `config.json`

- To enable __mixup augmentation__, under configuration setting `trainer_enhance`, set `"mixup": true` and `"mixup_alpha": n` where `n` in between 0 and 1, corresponding
to mixup augmentation parameter `alpha`.

- To enable __test-time augmentation__, under configuration setting `data_loader`, set `"p_aug": p` where p is probability of applying the image transformation (either one of time or frequency masking).

- To use __label smoothing loss function__, under configuration setting `loss` set `ce_labelsmoothing_loss`
