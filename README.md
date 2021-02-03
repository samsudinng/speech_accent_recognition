# Speech Accent Classification with Image Classifier
Accent classification from speech spectral images with image classifier


## 1. Features Image Extraction


### 1.1 Kaldi features (83-dim fbank + pitch)
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

### 1.2 Spectrogram features
The features are exctracted from the utterance .wav files from AESRC dataset. The extraction script is provided as Jupyter notebook at the moment, found in the folder `features_extraction\`. The utterance .wav files should be located in folder `audio` and features written to folder `features`. Modify the path as needed, or you can create a symbolic link to these folders (check the notebooks). Two features are available at the moment: log-spectrogram (200_dim, 10ms frames) and magnitude spectrogram (512-dim, 10ms frames based on VGGVox features, see following section).

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

#### 1.2.1 Log-spectrogram features (200-dim)

__Notebook__: __Logspec_Features_from_Audio.jpynb__

__Input__: utterances .wav files, test set listed in `features_extraction\train\utt2label` and `features_extraction\dev\utt2label`

__Output__: spectrograms in .pkl.gz files (splitted into 8 chunks due to large size), data type: uint8 (formatted as gray image pixel)

__Process__:

1. Calculate zscore scaler 

    - calculate and save StandardScaler based on the whole training set to perform spectrogram standardization to zero mean and unit standar deviation
    - can also be loaded from .pkl file if pre-computed

2. Audio to Spectrogram Image

    - pre-emphasis filtering
    - convert to spectrogram
    - zscore standardization
    - normalization to [0.0, 1.0] range
    - quantize to uint8 ([0, 255])
    - save all quantized features as dictionary (```all_spec```)
        - key  : utterance name (eg. ```AESRC2020-AMERICAN-ACCENT-G00473-G00473S1028```)
        - value: numpy array (uint8), shape = (F, T)
   - write to .pkl.gz


#### 1.2.2 VGGVox magnitude-spectrogram features (512-dim)

Adapted from:
https://github.com/samsudinng/VGGVox-PyTorch/blob/master/train.py

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

To train the model, try `python train.py -c config.json`. The training/validation cofiguration can be set in `config.json`. For details on folder structure of the
code and format of the configuration file, refer to [PyTorch Template Project readme file.](https://github.com/samsudinng/pytorch-template/blob/master/README.md)

Additional configuration has been added for mixup and test-time data augmentation implementation. The following can be enabled/disabled by setting some parameters
in `config.json`

- To enable __mixup augmentation__, under configuration setting `trainer_enhance`, set `"mixup": true` and `"mixup_alpha": n` where `n` in between 0 and 1, corresponding
to mixup augmentation parameter `alpha`.

- To enable __test-time augmentation__, under configuration setting `data_loader`, set `"p_aug": p` where p is probability of applying the image transformation (either one of time or frequency masking).

- To use __label smoothing loss function__, under configuration setting `loss` set `ce_labelsmoothing_loss`
