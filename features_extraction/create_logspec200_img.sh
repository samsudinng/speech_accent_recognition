#! /bin/sh

CREATE_TRAIN_IMG=true
CREATE_TEST_IMG=true
TRAINDIR="train_img"
DEVDIR="dev_img"
TESTDIR="test_img"


# Create training/dev spectrogram images from logspec200 features

#Train/dev img
if [ "$CREATE_TRAIN_IMG" = true ]; then
    if [ -d train_img ]; then
        echo "resetting train_img/"
        rm -r train_img
    fi
    mkdir train_img
    for i in {0..7}; do 
        mkdir train_img/$i; 
    done

    if [ -d dev_img ]; then
        echo "resetting dev_img/"
        rm -r dev_img
    fi
    mkdir dev_img
    for i in {0..7}; do 
        mkdir dev_img/$i; 
    done

    cp ../experiments/sbatch_template.sh create_img.sh
    echo "#SBATCH --job-name=logspec200_img"  >> create_img.sh
    echo "#SBATCH --output=out_logspec200_img.out"  >> create_img.sh
    echo "#SBATCH --error=err_logspec200_img.err"  >> create_img.sh
    echo conda activate accent >> create_img.sh
    echo python create_logspec_img.py >> create_img.sh
    sbatch create_img.sh
    rm create_img.sh
fi

#Test img
if [ "$CREATE_TEST_IMG" = true ]; then

    if [ -d test_img ]; then
        echo "resetting test_img/"
        rm -r test_img
    fi
    mkdir test_img
    for i in {0..7}; do
        mkdir test_img/$i;
    done

    cp ../experiments/sbatch_template.sh create_img.sh
    echo "#SBATCH --job-name=logspec200_testimg"  >> create_img.sh
    echo "#SBATCH --output=out_logspec200_testimg.out"  >> create_img.sh
    echo "#SBATCH --error=err_logspec200_testimg.err"  >> create_img.sh
    echo conda activate accent >> create_img.sh
    echo python create_logspec_test_img.py >> create_img.sh
    sbatch create_img.sh
    rm create_img.sh
fi


