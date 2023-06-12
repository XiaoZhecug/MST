"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import get_test_data_loaders
import time
from pre_post_procs.utils import magnitude2waveform, spectrum2magnitude
import soundfile as sf
from sklearn import manifold
import matplotlib.pyplot as plt
import librosa
import librosa.display


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/example.yaml', help="net configuration")
parser.add_argument('--input', type=str, default='dataset/spctra_quanti_original', help="input image path")
parser.add_argument('--output_folder', type=str, default='test_output', help="output image path")
parser.add_argument('--checkpoint', type=str, default='outputs/example/checkpoints/dataset_10sec_180000/gen_00180000.pt', help="checkpoint of pre-trained model")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=1, help="number of styles to sample")
parser.add_argument('--task_name',type=str, default='', help="task name, default will be config file name")


opts = parser.parse_args()

def npy2wav(npy, wav_name, config):
    wav_data = np.zeros((npy.shape[1], npy.shape[2]), dtype='float32')
    wav_data = npy[0]
    mag = spectrum2magnitude(wav_data, config)
    print('*'*30+'reconstructing waveform'+'*'*30)
    audio = magnitude2waveform(mag, config, None)
    print(audio.shape, audio.dtype)
    if not np.isfinite(audio).all():
        print('Error !!!\nThe audio is nan')
    if np.max(np.abs(audio)) > 0.0:
        # normalize the output audio
        norm_audio = audio/np.max(np.abs(audio))
    else:
        norm_audio = audio
    sf.write(wav_name, norm_audio, 22050, subtype='PCM_24')

def save_pianoroll(path, roll, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """ 
    im = Image.fromarray(roll)
    im = im.convert('RGB')
    img = np.asarray(im).copy()
    img = np.where(img<1, 255, img)
    im = Image.fromarray(img)
#   im.show()
    im = im.resize((im.size[0], im.size[1] * zoom))
    im.save(path)
    print(path)
    
def save_spc(path, roll):
    
    plt.figure(figsize=(10,10))
    librosa.display.specshow(roll, x_axis='time', hop_length=config['hop_length'])
    # plt.tight_layout(pad=-0.4, w_pad=0.0, h_pad=0.0)
    plt.savefig(path, bbox_inches='tight')
    plt.clf()
    
#     im = Image.fromarray(roll)
#     im = im.convert('RGB')
#     img = np.asarray(im).copy()
#     # img = np.where(img<1, 255, img)
#     # im = Image.fromarray(img)
# #   im.show()
#     # im = im.resize((im.size[0], im.size[1] * zoom))
#     im.save(path)
    print(path)
    
def show_graph(data, label_num):
    # X=np.identity(label_num)
    X=data
    y=np.argmax(np.eye(label_num), axis=1)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=500)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.  Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
     
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    
    marker= ['.', 'o','v','^','<','>','8','s','p','*','+']
    plt.figure(figsize=(8, 8))
    
    # print(X_norm.shape)
    # print(y)
    for i in range(X_norm.shape[0]):
        print(i)
        plt.scatter(X_norm[i, 0], X_norm[i, 1], s=60, color=plt.cm.tab20(int(i/200)), marker=marker[int(i/200)])
        plt.text(X_norm[i, 0]+0.015, X_norm[i, 1]-0.01, str(y[i]), color=plt.cm.tab20(int(i/200)), fontdict={'size': 10}, alpha= 1)          
        if (i==799):
            plt.text(0, 0+0.15, str("classical"), color=plt.cm.tab20(int(0)), fontdict={'size': 20}, alpha= 1)
            plt.text(0, 0+0.4, str("jazz"), color=plt.cm.tab20(int(1)), fontdict={'size': 20}, alpha= 1)
            plt.text(0.5, 0+0.8, str("fake_jazz"), color=plt.cm.tab20(int(2)), fontdict={'size': 20}, alpha= 1)
            plt.text(0.6, 0+0.6, str("fake_classical"), color=plt.cm.tab20(int(3)), fontdict={'size': 20}, alpha= 1)
            
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('X_tsne.pdf')
    plt.show()

print('*'*70)
print('config=', opts.config)
print('input=', opts.input)
print('output_folder=', opts.output_folder)
print('checkpoint=', opts.checkpoint)
print('a2b=', opts.a2b)
print('num_style=', opts.num_style)
print('seed=', opts.seed)
print('*'*70)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
style_dim = config['gen']['style_dim']
trainer = MUNIT_Trainer(config)

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()
# encode function
encode = trainer.gen_a.encode if opts.a2b == 1 else trainer.gen_b.encode
encode_inv = trainer.gen_b.encode if opts.a2b == 1 else trainer.gen_a.encode

style_encode = trainer.gen_b.encode if opts.a2b == 1 else trainer.gen_a.encode

# decode function
decode = trainer.gen_b.decode if opts.a2b == 1 else trainer.gen_a.decode # decode function


st = time.time()

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b == 1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

test_loader = get_test_data_loaders(config, opts.input, opts.a2b)

style_image = Variable(transforms(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda(), volatile=True) if opts.style != '' else None
iterations = opts.checkpoint[-11:-3]
if opts.task_name == '':
    task_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_folder, task_name+'_'+iterations)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

a2b_directory = os.path.join(output_directory, 'a2b')
b2a_directory = os.path.join(output_directory, 'b2a')
if not os.path.exists(a2b_directory):
    os.makedirs(a2b_directory)
if not os.path.exists(b2a_directory):
    os.makedirs(b2a_directory)

# Start testing
style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
for j in range(opts.num_style):
    style_npy_var = style_rand[j].cpu().numpy()
    if opts.a2b == 1:
        style_name = a2b_directory + '/style_code_' + str(j).zfill(2)
    else:
        style_name = b2a_directory + '/style_code_' + str(j).zfill(2)
    np.save(style_name, style_npy_var)
    
style_raw_list = np.empty((1, 8), dtype=float)   
style_trans_list = np.empty((1, 8), dtype=float)   
    
for it, (data) in enumerate(test_loader):
    images = data
    images = Variable(images.cuda())
    if opts.style != '':
        with torch.no_grad():
            _, style = style_encode(style_image)
    else:
        style = style_rand

    with torch.no_grad():
        content_raw, style_raw = encode(images)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content_raw, s)
            _, style_trans = encode_inv(outputs)
            
            x =  outputs[0].cpu().numpy()
            if opts.a2b == 1:
                npy_name_trans = a2b_directory + '/piece'+str(it).zfill(4) + '_transfed'
            else:
                npy_name_trans = b2a_directory + '/piece'+str(it).zfill(4) + '_transfed'
            npy_name_raw = npy_name_trans[:-9] + '_raw'
            print("save spctra numpy..", it)
            np.save(npy_name_raw, images[0].cpu().numpy())
            np.save(npy_name_trans, x)
            
            print("saved image file..", it)
            wav_name_raw = npy_name_raw + '.png'
            wav_name_trans = npy_name_trans + '.png'
            # print(images[0].cpu().numpy()[0, :].shape)
            # print(x[0, :].shape)
            image_raw = images[0].cpu().numpy()[0, :]
            image_trasfered = x[0, :]
            save_spc(wav_name_raw, images[0].cpu().numpy()[0, :])
            save_spc(wav_name_trans, x[0, :])
                        
            print("saved wav file..", it)
            wav_name_raw = npy_name_raw + '.wav'
            wav_name_trans = npy_name_trans + '.wav'
            npy2wav(images[0].cpu().numpy(), wav_name_raw, config)
            npy2wav(x, wav_name_trans, config)
            
            print("saved style numpy file..", it)

            style_raw_s =  style_raw[0].cpu().numpy()        
            style_raw_s = style_raw_s.reshape(1,8)
            style_raw_list = np.concatenate((style_raw_list, style_raw_s))
            style_raw_list_save = style_raw_list[1:,...]
            
            style_trans_s =  style_trans[0].cpu().numpy()                       
            style_trans_s = style_trans_s.reshape(1,8)
            style_trans_list = np.concatenate((style_trans_list, style_trans_s))
            style_trans_list_save = style_trans_list[1:,...]
            if(it == 14):
                npy_name_raw = npy_name_raw + '_style_raw_list'
                npy_name_trans = npy_name_trans + '_style_trans_list'
                np.save(npy_name_trans, style_trans_list_save)  
                np.save(npy_name_raw, style_raw_list_save) 
            
            
ed = time.time()
print("elapsed {0} seconds".format(ed-st))


# style_raw_list = np.empty((1, 8), dtype=float)   
# style_trans_list = np.empty((1, 8), dtype=float)        
# for it, (data) in enumerate(test_loader):
#     images = data
#     images = Variable(images.cuda())
#     with torch.no_grad():
#         content_raw, style_raw = encode(images)
#         s = style_rand.unsqueeze(0)
#         outputs = decode(content_raw, s)
#         _, style_trans = encode_a(outputs)
        
#         style_trans_c =  style_trans[0].cpu().numpy()        

            
          #style_trans_c = style_trans_c.reshape(1,8)
#         style_trans_list = np.concatenate((style_trans_list, style_trans_c))
#         style_trans_list_save = style_trans_list[1:,...]
        
#         style_raw_c =  style_raw[0].cpu().numpy()        
#         style_raw_c = style_raw_c.reshape(1,8)
#         style_raw_list = np.concatenate((style_raw_list, style_raw_c))
#         style_raw_list_save = style_raw_list[1:,...]
        
#         print("saved ", it)
#         if(it == 2399):
#             np.save(style_name + '_style_B_A_list', style_trans_list_save)  
#             np.save(style_name + '_style_B_raw_list', style_raw_list_save) 
#             break

# a = np.load('test_output/example_00100000/a2b/style_code_00_style_A_raw_list.npy')
# b = np.load('test_output/example_00100000/a2b/style_code_00_style_B_raw_list.npy')
# a_b = np.load('test_output/example_00100000/a2b/style_code_00_style_A_B_list.npy')
# b_a = np.load('test_output/example_00100000/a2b/style_code_00_style_B_A_list.npy')

# np.random.shuffle(a)
# np.random.shuffle(b)
# np.random.shuffle(a_b)
# np.random.shuffle(b_a)

# data = np.concatenate((a[0:200,...], b[0:200,...]))
# data = np.concatenate((data, a_b[0:200,...]))
# data = np.concatenate((data, b_a[0:200,...]))

# show_graph(data, 800)


