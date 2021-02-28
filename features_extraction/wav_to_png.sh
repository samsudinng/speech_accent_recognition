#!/bin/sh

##########################################
### SET THESE PATHS TO YOUR OWN
##########################################

### paths to the .wav and output feature files
ENABLE_WAV2FEATURE=true
TRAINWAVPATH="audio/"
TESTWAVPATH="testaudio/"
FPATH="features/"

### paths to the output .png images
ENABLE_FEATURE2PNG=true
TRAINIMGPATH="train_img/"
DEVIMGPATH="dev_img/"
TESTIMGPATH="test_img/"


###########################################
### SOME DETAILED SETTINGS
### *you probably won't need to touch these                
###########################################

### path to the metadata files
METAPATH="metadata/"

SEGMENTSIZE=300
FEATURE=logspec200
ZSCALERFILE="None"

### split the data set into chuncks
N_TRAIN_SPLIT=8
N_DEV_SPLIT=1
N_TEST_SPLIT=1

##Set to 1 to extract train/dev and test dataset
XTRACT_TRAIN=1
XTRACT_TEST=1

## set to 1 to delete wav files after features extraction
TO_DELETE_WAV=1
TO_DELETE_FEAT=0

###########################################
### CONVERT .WAV TO FEATURES
###########################################

if [ "$ENABLE_WAV2FEATURE" = true ]; then
    if [ -d $FPATH ]; then
        echo $FPATH" already exists, resetting the directory"
        rm -r $FPATH
    fi
    echo "Creating "$FPATH" directory"
    mkdir $FPATH

    echo "Converting .wav to features"
    python wav_to_features.py \
        --feature $FEATURE \
        --trainpath $TRAINWAVPATH \
        --testpath $TESTWAVPATH \
        --fpath $FPATH \
        --zscore $ZSCALERFILE \
        --metapath $METAPATH \
        --segment $SEGMENTSIZE \
        --trainsplit $N_TRAIN_SPLIT \
        --devsplit $N_DEV_SPLIT \
        --testsplit $N_TEST_SPLIT \
        --xtrain $XTRACT_TRAIN \
        --xtest $XTRACT_TEST \
        --delete $TO_DELETE_WAV
fi

###########################################
### CONVERT FEATURES TO PNG IMAGES
###########################################

if [ "$ENABLE_FEATURE2PNG" = true ]; then

    OUTPATH=$TRAINIMGPATH
    if [ -d $OUTPATH ]; then
        echo "resetting "$OUTPATH
        rm -r $OUTPATH
    fi
    echo "Creating "$OUTPATH" directory"
    mkdir $OUTPATH
    for i in {0..7}; do
        mkdir $OUTPATH/$i;
    done

    OUTPATH=$DEVIMGPATH
    if [ -d $OUTPATH ]; then
        echo "resetting "$OUTPATH
        rm -r $OUTPATH
    fi
    echo "Creating "$OUTPATH" directory"
    mkdir $OUTPATH
    for i in {0..7}; do
        mkdir $OUTPATH/$i;
    done

    OUTPATH=$TESTIMGPATH
    if [ -d $OUTPATH ]; then
        echo "resetting "$OUTPATH
        rm -r $OUTPATH
    fi
    echo "Creating "$OUTPATH" directory"
    mkdir $OUTPATH
    for i in {0..7}; do
        mkdir $OUTPATH/$i;
    done
    echo "Converting features to .png images"
    python features_to_png.py \
        --feature $FEATURE \
        --trainpath $TRAINIMGPATH \
        --devpath $DEVIMGPATH \
        --testpath $TESTIMGPATH \
        --fpath $FPATH \
        --metapath $METAPATH \
        --segment $SEGMENTSIZE \
        --trainsplit $N_TRAIN_SPLIT \
        --devsplit $N_DEV_SPLIT \
        --testsplit $N_TEST_SPLIT \
        --xtrain $XTRACT_TRAIN \
        --xtest $XTRACT_TEST \
        --delete $TO_DELETE_FEAT
fi
