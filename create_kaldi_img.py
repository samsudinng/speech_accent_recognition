import kaldiio
from kaldiio import ReadHelper
import os
import sys
import fileinput
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import math
from PIL import Image

#===========================parameters/variables to be set===============================
#FEATURE IMAGE PARAMETERS

#the dimension of each kaldi feature image segment (time direction). 
#Image size = (83 x segment_size) pixels
segment_size = 200


#TRAINING IMAGES

#folder containing train .ark and .scp
train_path = './kaldi_features/trainv2/deltafalse/'

#utt2accent
train_label_file ='./kaldi_features/trainv2/utt2accent'

#number of .scp file segments in the train set
n_train_feats_files = 32

#directory to save the resulting train images
train_dir = './train_img/'

#the original path to .ark files pointed in.scp, to be replaced with current path
old_train_path = '/home2/tungpham/espnet/esp_transformer/espnet-master/egs/contest_accentedEL/asr1/dump/trainv2/deltafalse/'


#DEV IMAGES

#folder containing dev .ark and .scp files
dev_path = './kaldi_features/devv2/deltafalse/'

#utt-accent
dev_label_file ='./kaldi_features/devv2/utt2accent'

#number of .scp file segments
n_dev_feats_files = 4

#directory to save the resulting dev images
dev_dir = './dev_img/'

#the original path to dev .ark files pointed in.scp, to be replaced with current path
old_dev_path = '/home2/tungpham/espnet/esp_transformer/espnet-master/egs/contest_accentedEL/asr1/dump/devv2/deltafalse/'

#==============================================================================================================================



def segment_nd_features(data, label, segment_size):
    '''
    Segment features into <segment_size> frames.
    Pad with 0 if data frames < segment_size

    Input:
    ------
        - data: shape is (Channel, Time, Freq)
        - label: accent label for the current utterance data
        - segment_size: length of each segment
    
    Return:
    -------
    Tuples of (number of segments, frames, segment labels, utterance label)
        - frames: ndarray of shape (N, C, F, T)
                    - N: number of segments
                    - C: number of input channels
                    - F: frequency index
                    - T: time index
        - segment labels: list of labels for each segments
                    - len(segment labels) == number of segments
    '''
    #  C, T, F
    nch = data.shape[0]
    time = data.shape[1]
    start, end = 0, segment_size
    num_segs = math.ceil(time / segment_size) # number of segments of each utterance
    data_tot = []
    sf = 0
    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        
        # Do padding
        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
                #data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant",
                #constant_values=((-80,-80),(-80,-80)))
            data_pad.append(data_ch)

        data_pad = np.array(data_pad)
        
        # Stack
        data_tot.append(data_pad)
        
        # Update variables
        start = end
        end = min(time, end + segment_size)
    
    data_tot = np.stack(data_tot)
    utt_label = label
    segment_labels = [label] * num_segs
    
    #Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0,1,3,2)

    return (num_segs, data_tot, segment_labels, utt_label)



def update_scp_path(data_path, old_data_path, n_feat_files):

    # Update the kaldi features path in each .scp file

    total_utt = 0
    for fidx in range(n_feat_files ):
        
        scp_in = f'{data_path}feats.{fidx+1}.scp'
        scp_out= f'{data_path}feats_mod.{fidx+1}.scp'
        
        #get the original lines of feats.scp
        with open(scp_in) as ff:
            Lines = ff.readlines()
        n_lines = len(Lines)
        
        #modify the file path
        outFile = open( scp_out, 'w' )
        n_mod_lines = 0
        for line in Lines:
            if old_data_path not in line :
                raise valueError('Match Not Found!!')
            outFile.write( line.replace( old_data_path, data_path ) )
            assert len(line.split(' ')) == 2
            n_mod_lines += 1
        
        outFile.close()

        assert n_mod_lines == n_lines
        total_utt += n_lines
    
    return total_utt


def convert_to_image(data_path, data_labels, n_feats_files, scaler, img_dir):
    """
    Perform the following:
        1. Read the kaldi features matrix, 
        2. scale data to [0,1], 
        3. convert values to 8-bit pixel, segment, 
        4. save as .png image
    """
    data_labels['num_segments']=""

    for nf in range(n_feats_files):

        with ReadHelper(f'scp:{data_path}feats_mod.{nf+1}.scp') as reader:
            for key, mat in reader:

                label = data_labels.loc[key,'label']
                #mat: T, F
                #normalize [0,1] and fix orienation of freq axis
                features = scaler.transform(mat)
                features = np.fliplr(features)

                #clip and convert to 8-bit pixels
                features = np.clip(features, 0.0, 1.0)
                features = (features*255).astype(np.uint8)

                #segment
                features = np.expand_dims(features, 0)
                features_segmented = segment_nd_features(features, label, segment_size)

                num_segments, segments, _, _ = features_segmented

                for idx, img in enumerate(segments):
                    pil_img = Image.fromarray(np.squeeze(img,0))
                    pil_img.save(f'{img_dir}{label}/{key}_{idx}.png')

                data_labels.loc[key,'num_segments'] = num_segments

    return data_labels






#Update train and dev datapath in the .scp files
num_train_utt = update_scp_path(train_path, old_train_path, n_train_feats_files)
num_dev_utt = update_scp_path(dev_path, old_dev_path, n_dev_feats_files)


#Read the label data into dataframe
train_labels = pd.read_csv(train_label_file, header=None, delim_whitespace=True,index_col=0)
train_labels.columns = ['label']
dev_labels = pd.read_csv(dev_label_file, header=None, delim_whitespace=True,index_col=0)
dev_labels.columns = ['label']


#Read the training features and fit MinMaxScaler over the whole training data
train_scaler = MinMaxScaler()
nsamples = 0
for nf in range(n_train_feats_files): 
    features = []
    with ReadHelper(f'scp:{train_path}feats_mod.{nf+1}.scp') as reader:
        for key, mat in reader:
            features.append(mat)
    nsamples += len(features)
    features=np.concatenate(features, axis=0)
    train_scaler.partial_fit(features)


#Segment the features image and save the images respective label directory
train_labels = convert_to_image(train_path, train_labels, n_train_feats_files, train_scaler, train_dir)
dev_labels = convert_to_image(dev_path, dev_labels, n_dev_feats_files, train_scaler, dev_dir) 


#save the label dataframe
train_labels.to_pickle(f'{train_dir}train_labels.pkl')
dev_labels.to_pickle(f'{dev_dir}dev_labels.pkl')




