import numpy as np
import pandas as pd
import pickle, gzip
from tqdm import tqdm
from PIL import Image
import math
import gc
#===========================parameters/variables to be set===============================
#FEATURE IMAGE PARAMETERS

#the dimension of each logspec feature image segment (time direction). 
#Image size = (200 x segment_size) pixels
segment_size = 300


#TRAIN IMAGES

#folder containing .pkl.gz spectrogram images
train_feature_dir = '../logspec_features/'
dev_feature_dir = '../logspec_features/'

#utt2accent
train_label_file ='../kaldi_features/trainv2/utt2accent'
dev_label_file ='../kaldi_features/devv2/utt2accent'

#number of .pkl.gz file segments in the train set
n_train_feats_files = 8
n_dev_feats_files = 1

#directory to save the resulting train images
train_dir = '../train_img/'
dev_dir = '../dev_img/'

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



def convert_to_image(allspec, data_labels, img_dir):
    """
    Perform the following:
        1. convert values to 8-bit pixel, segment, 
        2. save as .png image
    """
    #data_labels['num_segments']=np.nan

    for utt,logspec in allspec.items():
        #print(utt)
        #get the utterance label
        label = data_labels.loc[utt,'label']        

        #segment
        logspec = np.expand_dims(logspec, 0).transpose(0,2,1) #(1,T,F)
        logspec_segmented = segment_nd_features(logspec, label, segment_size)

        num_segments, segments, _, _ = logspec_segmented

        for idx, img in enumerate(segments):
            pil_img = Image.fromarray(np.squeeze(img,0))
            pil_img.save(f'{img_dir}{label}/{utt}_{idx}.png')

        data_labels.loc[utt,'num_segments'] = num_segments

    return data_labels


#Read the label data into dataframe
train_labels = pd.read_csv(train_label_file, header=None, delim_whitespace=True,index_col=0)
train_labels.columns = ['label']
train_labels['num_segments']=np.nan
dev_labels = pd.read_csv(dev_label_file, header=None, delim_whitespace=True,index_col=0)
dev_labels.columns = ['label']
dev_labels['num_segments']=np.nan


#Read the training features, segment and put in train_img folder
#   Format of .pkl.gz content: {utterance:log_spectrogram}
#       - utterance: utterance name as in utt2accent, eg.AESRC2020-AMERICAN-ACCENT-G00473-G00473S1001
#       - log_spectrogram: (F, T), standardized and quantized to uint8

for nf in range(n_train_feats_files):
    filename = f'{train_feature_dir}train_alexnet_{nf}.pkl.gz'
    print(filename)
    with gzip.open(filename, 'rb') as f:
        allspec = pickle.load(f)
    
    train_labels = convert_to_image(allspec, train_labels, train_dir)        

    del allspec
    gc.collect()

#Read the dev features, segment and put in dev_img folder
for nf in range(n_dev_feats_files):
    filename = f'{dev_feature_dir}dev_alexnet_{nf}.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        allspec = pickle.load(f)
    dev_labels = convert_to_image(allspec, dev_labels, dev_dir) 

    del allspec
    gc.collect()

#make sure all files are processed
#assert train_labels.num_segments.isna().any().item() == False
#assert dev_labels.num_segments.isna().any().item() == False

#save the label dataframe
#train_labels.to_pickle(f'{train_dir}train_labels.pkl')
dev_labels.to_pickle(f'{dev_dir}dev_labels.pkl')




