#!/bin/bash

##########################################
### SET THESE PATHS TO YOUR OWN
##########################################

### paths to the .wav files
TRAINWAVPATH="audio/"
TESTWAVPATH="testaudio/"
ZSCALERPATH="None"

### path to the feature files
FPATH="features/"

### path to the metadata files
METAPATH="metadata/"

### paths to the output .png images
TRAINIMGPATH="train_img"
DEVIMGPATH="dev_img"
TESTIMGPATH="test_img"


###########################################
### SOME DETAILED SETTINGS
### *you probably won't need to touch these                
###########################################

SEGMENTSIZE=300
FEATURE=logspec200

### split the data set into chuncks
N_TRAIN_SPLIT=8
N_DEV_SPLIT=1
N_TEST_SPLIT=1



###########################################
### CONVERT .WAV TO FEATURES                
###########################################

python wav_to_features.py \
    --feature $FEATURE \
    --trainpath $TRAINWAVPATH \
    --testpath $TESTWAVPATH \
    --fpath $FPATH \
    --zscore $ZSCALERPATH \
    --metapath $METAPATH \
    --segment $SEGMENTSIZE \
    --trainsplit $N_TRAIN_SPLIT \
    --devsplit $N_DEV_SPLIT \
    --testsplit $N_TEST_SPLIT


###########################################
### CONVERT FEATURES TO PNG IMAGES                
###########################################

OUTPATH=$TRAINIMGPATH
if [ -d $OUTPATH ]; then
    echo "resetting "$OUTPATH
    rm -r $OUTPATH
fi
mkdir $OUTPATH
for i in {0..7}; do
    mkdir $OUTPATH/$i;
done

OUTPATH=$DEVIMGPATH
if [ -d $OUTPATH ]; then
    echo "resetting "$OUTPATH
    rm -r $OUTPATH
fi
mkdir $OUTPATH
for i in {0..7}; do
    mkdir $OUTPATH/$i;
done

OUTPATH=$TESTIMGPATH
if [ -d $OUTPATH ]; then
    echo "resetting "$OUTPATH
    rm -r $OUTPATH
fi
mkdir $OUTPATH
for i in {0..7}; do
    mkdir $OUTPATH/$i;
done

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
    --testsplit $N_TEST_SPLIT
