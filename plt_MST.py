"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# from torch.utils.serialization import load_lua

import torch
import os
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_loss
# slerp
# get_slerp_interp
# get_model_list            : get model list for resume
# get_scheduler
# weights_init



def write2spec(outputs, display_image_num, file_name, config, feature_type):
    # file_name = ".../a2b_test_spec_"
    # only handles a channel
    # so expected to have a list of [4,256,256] * 4
    fig_name = file_name + '.pdf'
    #print('writing to ', fig_name)
    row = len(outputs)
    col = 4 # it should be display_image_num, but train_current will crash so it's a fixed '4'
    
    if row==3: # UNIT
        #names = ['x', 'recon', 'trans']
        plt.figure(num=1, figsize=(16,10)) # UNIT
    elif row==4: # MUNIT
        #names = ['x', 'recon', 'trans_fix', 'trans_rand']
        plt.figure(num=1, figsize=(16,13.33)) # MUNIT
    
    plt.clf()
    
    idx_row = -1
    for img in outputs:
        idx_row += 1
        idx_col = 0
        for idx in range(4):
            idx_col += 1
            x = img[idx] # img[idx] is a [256, 256]
            # now it's time to tranform it to dB and plot
            # but now default is using a pseudo-dB unit
            # because no power^0.3, it's too large to learn
            plt.subplot(row, col, idx_row*col+idx_col)
            if feature_type=='ceps':
                # the y-axis for cepstrums is quefrency
                librosa.display.specshow(x, x_axis='time', hop_length=config['hop_length'])
            else: # it's spec or diff_spec or spec_enve
                if ('is_mel' in config) and (config['is_mel']==True):
                    librosa.display.specshow(x, x_axis='time', y_axis='mel', hop_length=config['hop_length'])
                else:
                    librosa.display.specshow(x, x_axis='time', y_axis='linear', hop_length=config['hop_length'])
    plt.tight_layout(pad=-0.4, w_pad=0.0, h_pad=0.0)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.clf()
    

data_raw = np.load('piece0012_raw.npy')
data_transfed = np.load('piece0012_transfed.npy')

figure1 = data_raw[0,...,...] 
figure2 = data_raw[1,...,...] 
figure3 = data_raw[2,...,...] 
figure4 = data_raw[3,...,...] 

figure5 = data_transfed[0,...,...] 
figure6 = data_transfed[1,...,...] 
figure7 = data_transfed[2,...,...] 
figure8 = data_transfed[3,...,...] 




