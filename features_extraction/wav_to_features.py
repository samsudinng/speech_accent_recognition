import pandas as pd
import numpy as np
import librosa
import pickle, gzip
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
import gc
import argparse
import sys
import warnings
import os

#############################
# CONSTANTS AND LOOKUP TABLES 
#############################

#to map accent from Kaldi utt2accent to the corresponding accent folder
accent_dir = {
    'BRITISH'   : 'UK',       
    'AMERICAN'  : 'US',
    'PORTUGUESE': 'PT',  
    'KOREAN'    : 'KR',
    'JAPANESE'  : 'JPN',
    'RUSSIAN'   : 'RU',
    'CHINESE'   : 'CHN',
    'INDIAN'    : 'IND'   
}

#various spectrogram parameters
spectrogram_params = {
    'logspec200': {    # this is the same setting as SER features
        'window'        : 'hamming',
        'sampling_freq' : 16000,
        'win_len'       : 40, #msec
        'hop_len'       : 10, #msec
        'ndft'          : 800,
        'nfreq'         : 200,
    }
}


def get_labels(metadata):
    """
    Read the metadata and create dataframe consisting of utterance names ('utt')
        and labels ('label','age','gender' etc)
    """

    first = 0
    for m in metadata:
        temp = pd.read_csv(metadata[m], names = ['utt',m], header=None, 
                                delim_whitespace=True,index_col='utt')
        
        if first == 0:
            labels = temp
            first = 1
        else:
            labels[m] = temp
    labels.reset_index(inplace=True)          
    return labels

def get_wavfile_paths(labels, wavdir):
    """
    Get the full path (/path/to/utt_filename.wav) of the wav files and append
        to label dataframe
    """

    #extract the file names from `utt`
    accent = labels.utt.str.split(pat='-',expand=True).iloc[:,1].map(accent_dir).tolist()
    speaker = labels.utt.str.split(pat='-',expand=True).iloc[:,3].tolist()
    utt  = labels.utt.str.split(pat='-',expand=True).iloc[:,4].tolist()
    wavfile = list(zip(accent,speaker,utt))
    labels['wavfile'] = [wavdir+'/'.join(s)+'.wav' for s in wavfile]

    return labels

def get_wavfile_paths_from_scp(labels, wav_scp, wavdir):
    """
    Get the full path (/path/to/utt_filename.wav) of the wav files from wav.scp files and append
          to label dataframe
    """

    #get the file path from wav.scp
    with open(wav_scp) as ff:
       Lines = ff.readlines()
       n_lines = len(Lines)

    utt, wavfile = zip(*[ln.rstrip().split('  ') for ln in Lines])
    wavfile = [wavdir+f.split('/')[-1] for f in wavfile]
    
    wav_df = pd.DataFrame(zip(utt,wavfile),columns=['utt','wavfile']).set_index('utt')

    labels.set_index('utt',inplace=True)
    labels['wavfile'] = wav_df
    labels.reset_index(inplace=True)

    return labels


def calculate_log_spectrogram(x, sr, params):
    
    #convert msec to samples
    win_len = int(params['win_len'] * sr / 1000)
    hop_len = int(params['hop_len'] * sr / 1000)
    
    # Apply pre-emphasis filter
    x = librosa.effects.preemphasis(x, zi = [0.0])

    #calculate log-spectrogram
    spec = np.abs(librosa.stft(x, n_fft=params['ndft'],hop_length=hop_len,
                                        win_length=win_len,
                                        window=params['window']))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)
    
    #extract the required frequency bins
    spec = spec[:params['nfreq']]
    
    
    return spec #(F, T)


def get_data_range(x):
    
    max_val = np.max(x.flatten())
    min_val = np.min(x.flatten())
    
    return max_val, min_val

    
def calculate_zscore_scaler(train_labels,spec_params):
    
    zscore_scaler = StandardScaler() #z-score standardize the spectrogram
    calculate_every = 200
    
    all_spec=[]
    corrupted_files = []
    
    for row, col in tqdm(train_labels.iterrows()):
        wavfilename = col['wavfile']
    
        #Read utterance wav file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, sr = librosa.load(wavfilename, sr=None)
        except:
            corrupted_files.append(wavfilename)
            continue
            
        assert sr == spec_params['sampling_freq']
    
        #Calculate log spectrogram
        log_spec = calculate_log_spectrogram(x, sr, spec_params)
        all_spec.append(log_spec.T)
    
        #Calculate (partial) scaling factors based on overall training set
        if row > 0 and row % (calculate_every-1) == 0:
            zscore_scaler.partial_fit(np.vstack(all_spec))   
            all_spec[:] = []
    
    #last block
    zscore_scaler.partial_fit(np.vstack(all_spec))
    
    return zscore_scaler, corrupted_files


def extract_feature_image(df, spec_params, zscore_scaler, data_range,to_delete):
    
    all_spec={}
    minmax_scaler = MinMaxScaler(feature_range=(0,1))
    
    corrupted_files = []    
    for row, col in tqdm(df.iterrows()):
        utt = col['utt']
        label = col['label']
        wavfilename = col['wavfile']
    
        #Read utterance wav file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, sr = librosa.load(wavfilename, sr=None)
        except:
            corrupted_files.append(wavfilename)
            continue
            
        assert sr == spec_params['sampling_freq']
    
        #Calculate log spectrogram
        log_spec = calculate_log_spectrogram(x, sr, spec_params)
    
        #Standardization
        vmax,vmin = get_data_range(log_spec)
        data_range['max_ori'].append(vmax)
        data_range['min_ori'].append(vmin)
        log_spec = zscore_scaler.transform(log_spec.T)
        vmax,vmin = get_data_range(log_spec)
        data_range['max_std'].append(vmax)
        data_range['min_std'].append(vmin)
    
        #Convert range to [0, 1] and quantize to uint8
        minmax_scaler.fit(log_spec)
        log_spec = minmax_scaler.transform(log_spec)
        log_spec = (log_spec*255).astype(np.uint8)
        
        #Append to spectrogram dictionary
        all_spec[utt] = np.flipud(log_spec.T)
        
        #Delete wav files
        if to_delete == 1:
            os.remove(wavfilename)

    return all_spec, data_range, corrupted_files

    

def main(args):

    #PATHS
    spec_params = spectrogram_params[args.feature]
    wavdir      = args.trainpath
    testwavdir  = args.testpath
    outdir      = args.fpath
    metapath    = args.metapath
    to_delete   = args.delete
    xtract_train= args.xtrain
    xtract_test = args.xtest

    #METADATA
    train_metadata = {
                        'label'  : metapath+'train_utt2accent',
                        'gender' : metapath+'train_utt2sex',
                        'age'    : metapath+'train_utt2age'
                     }

    dev_metadata   = {
                         'label'  : metapath+'dev_utt2accent',
                         'gender' : metapath+'dev_utt2sex',
                         'age'    : metapath+'dev_utt2age'
                      }

    test_metadata  = {
                         'label'  : metapath+'test_utt2accent',
                     }

    test_wav_scp   = metapath+'test_wav.scp'
    
    #DATASET SPLIT
    n_train_split = args.trainsplit
    n_dev_split   = args.devsplit
    n_test_split  = args.testsplit

    #get utt and labels
    train_labels = get_labels(train_metadata)
    dev_labels   = get_labels(dev_metadata)
    test_labels  = get_labels(test_metadata)

    #get full paths to wav files
    train_df = get_wavfile_paths(train_labels, wavdir)
    dev_df   = get_wavfile_paths(dev_labels, wavdir)
    test_df  = get_wavfile_paths_from_scp(test_labels, test_wav_scp, testwavdir)
    
    
    # Calculate zscore scaling from training dataset
    if xtract_train == 1:
        if args.zscore == 'None':
            print('Calculating zscore scaler from training data')
            zscore_scaler,corrupted_files = calculate_zscore_scaler(train_df,spec_params)
            print(f'Corrupted files: {len(corrupted_files)}\n{corrupted_files}\n')
            with open(f'{outdir}zscore_scaler_{args.feature}.pkl','wb') as fi:
                pickle.dump(zscore_scaler, fi)
        else:
            with open(args.zscore,'rb') as fi:
                zscore_scaler = pickle.load(fi)
    elif xtract_test == 1:
        assert args.zscore != 'None'    
        with open(args.zscore,'rb') as fi:
            zscore_scaler = pickle.load(fi)
    
    # ==========================================================================
    # Extract features from train set
    # ==========================================================================
    
    if xtract_train == 1:
        print('Extracting features from training data')
        data_range = defaultdict(list)

        # split the training data into chuncks (due to huge memory size)
        nsplit = n_train_split 
        split_size = int(np.floor(len(train_df)/nsplit))
        split_start = [split_size*idx for idx in list(range(0,nsplit,1))]
        split_end = [split_size*idx for idx in list(range(1,nsplit,1))]
        split_end.append(len(train_df))
        split_idx = zip(split_start, split_end)

        corrupted_files=[]
        for chunk, idx in enumerate(split_idx):
            start,end = idx
            print(f'Chunk {chunk+1}/{nsplit}\n')
            all_spec, data_range,corrupted = extract_feature_image(train_df.iloc[start:end,:], spec_params, zscore_scaler, data_range, to_delete)
            corrupted_files.extend(corrupted)
            pickle.dump( all_spec, gzip.open( f'{outdir}train_{args.feature}_{chunk}.pkl.gz',   'wb' ) )
    
            del all_spec
            gc.collect()

        gmin = min(data_range['min_ori'])
        gmax = max(data_range['max_ori'])
        print(f'Data range original    : min: {gmin} - max: {gmax}')
    
        gmin = min(data_range['min_std'])
        gmax = max(data_range['max_std'])
        print(f'Data range standardized: min: {gmin} - max: {gmax}')

        print(f'Corrupted files: {len(corrupted_files)}\n{corrupted_files}\n')
    
        del data_range
        gc.collect()
    

    # ==========================================================================
    # Extract features from dev set
    # ==========================================================================
    
    if xtract_train == 1:
        print('Extracting features from dev data')
        data_range = defaultdict(list)

        # split the training data into chuncks (due to huge memory size)
        nsplit = n_dev_split
        split_size = int(np.floor(len(dev_df)/nsplit))
        split_start = [split_size*idx for idx in list(range(0,nsplit,1))]
        split_end = [split_size*idx for idx in list(range(1,nsplit,1))]
        split_end.append(len(dev_df))
        split_idx = zip(split_start, split_end)
        corrupted_files = []
        for chunk, idx in enumerate(split_idx):
            start,end = idx
            print(f'Chunk {chunk+1}/{nsplit}\n')         
            all_spec, data_range,corrupted = extract_feature_image(dev_df.iloc[start:end,:], spec_params, zscore_scaler, data_range, to_delete)
            corrupted_files.extend(corrupted)
            pickle.dump( all_spec, gzip.open( f'{outdir}dev_{args.feature}_{chunk}.pkl.gz',   'wb' ) )

            del all_spec
            gc.collect()

        gmin = min(data_range['min_ori'])
        gmax = max(data_range['max_ori'])
        print(f'Data range original    : min: {gmin} - max: {gmax}')

        gmin = min(data_range['min_std'])
        gmax = max(data_range['max_std'])
        print(f'Data range standardized: min: {gmin} - max: {gmax}')

        print(f'Corrupted files: {len(corrupted_files)}\n{corrupted_files}\n')
        del data_range
        gc.collect()


    # ==========================================================================
    # Extract features from test set
    # ==========================================================================

    if xtract_test == 1:
        print('Extracting features from test data')
        data_range = defaultdict(list)
     
        # split the training data into chuncks (due to huge memory size)
        nsplit = n_test_split
        split_size = int(np.floor(len(test_df)/nsplit))
        split_start = [split_size*idx for idx in list(range(0,nsplit,1))]
        split_end = [split_size*idx for idx in list(range(1,nsplit,1))]
        split_end.append(len(test_df))
        split_idx = zip(split_start, split_end)

        corrupted_files=[]
        for chunk, idx in enumerate(split_idx):
            start,end = idx
            print(f'Chunk {chunk+1}/{nsplit}\n')
            all_spec, data_range,corrupted = extract_feature_image(test_df.iloc[start:end,:], spec_params, zscore_scaler, data_range, to_delete)
            corrupted_files.extend(corrupted)
            pickle.dump( all_spec, gzip.open( f'{outdir}test_{args.feature}_{chunk}.pkl.gz',   'wb' ) )
         
            del all_spec
            gc.collect()
      
        gmin = min(data_range['min_ori'])
        gmax = max(data_range['max_ori'])
        print(f'Data range original    : min: {gmin} - max: {gmax}')
     
        gmin = min(data_range['min_std'])
        gmax = max(data_range['max_std'])
        print(f'Data range standardized: min: {gmin} - max: {gmax}')

        print(f'Corrupted files: {len(corrupted_files)}\n{corrupted_files}\n')
    
        del data_range
        gc.collect()

    print(f'\n{args.feature} features extraction done!')
    

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .wav utterances into spectrogram-based features (default: logspec200)")

    parser.add_argument('--feature', type=str, default='logspec200',
        help='Features to be extracted')
    parser.add_argument('--trainpath', type=str, default='audio/',
         help='path to train/dev .wav files')
    parser.add_argument('--testpath', type=str, default='testaudio/',
          help='path to test .wav files')
    parser.add_argument('--fpath', type=str, default='features/',
         help='path to save feature files')
    parser.add_argument('--zscore', type=str, default=None, 
         help='path to save zscore StandardScaler files')
    parser.add_argument('--metapath', type=str, default='metadata/',
         help='path to metadata files')
    parser.add_argument('--segment', type=int, default=300,
         help='segment size, set to -1 for non-segmented')
    parser.add_argument('--trainsplit', type=int, default=8,
         help='number of train set split')
    parser.add_argument('--devsplit', type=int, default=1,
         help='number of dev set split')
    parser.add_argument('--testsplit', type=int, default=1,
         help='number of test set split')
    parser.add_argument('--xtrain', type=int, default=1,
         help='set 1 to extract train\dev')
    parser.add_argument('--xtest', type=int, default=1,
         help='set 1 to extract test')
    parser.add_argument('--delete', type=int, default=0,
         help='set to 1 to delete wav files after conversion')    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
