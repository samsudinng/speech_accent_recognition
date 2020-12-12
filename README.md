# Speech Accent Classification with Image Classifier
Accent classification from speech spectral images with image classifier


## 1. Features Image Extraction

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

## 2. Training/Validation

To train the model, try `python train.py -c config.json`. The training/validation cofiguration can be set in `config.json`. For details on folder structure of the
code and format of the configuration file, refer to [PyTorch Template Project readme file.](https://github.com/samsudinng/pytorch-template/blob/master/README.md)

Additional configuration has been added for mixup and test-time data augmentation implementation. The following can be enabled/disabled by setting some parameters
in `config.json`

- To enable mixup augmentation, under configuration setting `trainer_enhance`, set `"mixup": true` and `"mixup_alpha": n` where `n` in between 0 and 1, corresponding
to mixup augmentation parameter `alpha`.

- To enable test-time augmentation, under configuration setting `data_loader`, set `"p_aug": p` where p is probability of applying the image transformation (either
one of GaussNoise, RandomBrightnessContrast, and ShiftScaleRotate from `albumentations` package).
