# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:32:11 2021

@author: Ali
"""

###############################################################################
import os
import time
import sys
import multiprocessing as mp
from multiprocessing import dummy as multiprocessing
import tensorflow as tf
import logging
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import socket
#from inception_utils import *
#import inception
import h5py
#slim = tf.contrib.slim

#ali modified
from PIL import Image
#import vgg
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inception
from tensorflow.keras.applications.vgg16 import VGG16,decode_predictions, preprocess_input
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, Dense


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

"""fix for issue: cuDNN failed to initialize"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('...GPU set_memory_growth successfully set...')

else:
    print('...GPU set_memory_growth not set...')


# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:17:29 2021

@author: Ali
"""
import math
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    
        
default= False # True for inception/imagenet otherwise, VGG/CUB
top_only = False #if True, apply method on top_conv filters only otherwise apply on all filters
if default:    
    DATA_DIR = 'G:/ILSVRC2012_img_val'
    MEM_DIR = './results'
    CHECKPOINT = 'imagenet.hdf5'#'inception_v3.ckpt'
    labels_path = os.path.join('imagenet_labels.txt')
else:
    DATA_DIR = 'G:/CUB_200_2011/CUB_200_2011/train_test_split/test'
    MEM_DIR = './results'
    CHECKPOINT = 'model_fine_tune_epoch_150.hdf5'#'inception_v3.ckpt'
    labels_path = os.path.join('CUB_labels.txt')    

BATCH_SIZE = 16 #128 default
TIME_START = time.time()

val_labels = tf.io.gfile.GFile(labels_path).read().splitlines()

#@tf.function
def remove_players(model, players):
    '''remove selected players in the Inception-v3 network.'''
    if isinstance(players, str):
        players = [players]
    for player in players:
        variables = layer_dic['_'.join(player.split('_')[:-1])]
        #var_vals = model.sess.run(variables)
        #var_vals = variables
        for var in variables:
            var_copy=var.numpy()
            if 'variance' in var.name:
                var_copy[..., int(player.split('_')[-1])] = 1.
            elif 'beta' in var.name:
                pass
            elif 'bias' in var.name:#ali modified. TODO: what to do with bias??
                pass
            else:
                var_copy[..., int(player.split('_')[-1])] = 0.
            #var.load(var, model.sess)
            var.assign(var_copy) #verified that it sets weights to zero in the model
        

def return_player_output(model, player):
    '''The output of a filter.'''
    layer = '_'.join(player.split('_')[:-1])
    layer = '/'.join(layer.split('/')[1:])
    unit = int(player.split('_')[-1])
    return model.ends[layer][..., unit]


def get_model_acc(model, actual_test_gen):
    pred_probs= model.predict(actual_test_gen,verbose=1)
    
    pred_classes = np.argmax(pred_probs,1)
    #actual_classes = np.argmax(test_gen.classes,1)
    actual_classes = actual_test_gen.classes
    print(confusion_matrix(actual_classes,pred_classes))
    print(classification_report(actual_classes,pred_classes,digits=4))
    
    acc = np.sum(pred_classes==actual_classes)/actual_test_gen.n
    return acc
    

def eval_model(
    model, 
    players,
    chosen_players=None,
    c=None, 
    actual_test_gen = None
):
    model.load_weights(filepath=CHECKPOINT)
    
    old_val = get_model_acc(model, actual_test_gen)
    if c is None:
        c = {i: np.array([i]) for i in range(len(players))}
    elif not isinstance(c, dict):
        c = {i: np.where(c==i)[0] for i in set(c)}
    if chosen_players is None:
        chosen_players = np.arange(len(c.keys()))
    idxs = np.random.permutation(len(c.keys()))
    #print(idxs)
    
    for n, idx in enumerate(idxs[::-1]):
        if idx not in chosen_players:
            pass
        else:
            #old_val = None
            remove_players(model, players[c[idx]])        
            
    val = get_model_acc(model, actual_test_gen)
    
    print("Original model accuracy:", old_val)
    print("Disabled top ",len(chosen_players), " filters")
    print("New model accuracy:", val)
    
    return old_val, val

def load_vgg():    
    print('using VGG16 imagenet weights for CUB200 dataset')
    vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
    base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    x =  MaxPool2D()(base_model.output)
    mean_fmap = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(mean_fmap)
    x = Dense(200,activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs= [x])#, base_model.output])
    model.load_weights(CHECKPOINT)
    model.summary()
    return model

np.random.seed(0)
key = sys.argv[1] #Class name. Use 'all' for overll performance.
model_scope = 'InceptionV3'
metric = sys.argv[2] #metric one of accuracy, binary, xe_loss.
num_images = int(sys.argv[3]) #Number of validation images.
bound = 'Bernstein'
truncation = 0.2
max_sample_size = 128
adversarial = (sys.argv[4] == 'True') #If True, computes contributions for adversarial setting.
time.sleep(10 * np.random.random())
## Experiment Directory
if default:
    experiment_dir = os.path.join(MEM_DIR, 'NShap/inceptionv3/{}_{}_new'.format(metric, key))
else:
    if top_only:
        experiment_dir = os.path.join(MEM_DIR, 'NShap/vgg/{}_{}_top_layer_new'.format(metric, key))    
    else:
        experiment_dir = os.path.join(MEM_DIR, 'NShap/vgg/{}_{}_all_layer_new'.format(metric, key))    
if not tf.io.gfile.exists(experiment_dir):
    tf.io.gfile.makedirs(experiment_dir)
if max_sample_size is None or max_sample_size > num_images:
    max_sample_size = num_images
experiment_name = 'cb_{}_{}_{}'.format(bound, truncation, max_sample_size)
if adversarial:
    experiment_name = 'ADV' + experiment_name
cb_dir = os.path.join(experiment_dir, experiment_name)
if not tf.io.gfile.exists(cb_dir):
    tf.io.gfile.makedirs(cb_dir)
print(cb_dir)
#tf.reset_default_graph()
# model = inception.inpcetion_instance(checkpoint=CHECKPOINT)
#model = inception.inpcetion_instance()#checkpoint=CHECKPOINT#
if default:
    model = inception(weights=CHECKPOINT)
else:
    model = load_vgg()
model_variables = model.variables#tf.global_variables(scope=model_scope)
convs = ['/'.join(k.name.split('/')[:-1]) for k in model_variables if 'conv'
         in k.name and 'Aux' not in k.name and 'Logits' not in k.name]

#select top conv only (should be 2048 filters combined from 5? layers) -- yes verified that it is givig 2048 filters
if top_only:
    if default:
        convs=convs[-6:]
    else:
        convs=[convs[-1]]



layer_dic = {conv: [var for var in model_variables if conv == var.name.split('/')[:-1][0]]
             for conv in convs}
if tf.io.gfile.exists(os.path.join(experiment_dir, 'players.txt')) and False:
    players = open(os.path.join(
        experiment_dir, 'players.txt')).read().split(',')
    players = np.array(players)
else:
    players = []
    var_dic = {var.name: var for var in model_variables}
    for conv in layer_dic.keys():
        players.append(['{}_{}'.format(conv, i) for i in
                        range(var_dic[conv + '/kernel:0'].shape[-1])])
    players = np.sort(np.concatenate(players))
    open(os.path.join(experiment_dir, 'players.txt'), 'w').write(
        ','.join(players))
if metric == 'accuracy':
    base_value = 1./1000
elif metric == 'xe_loss':
    base_value = -np.log(1000)
elif metric == 'binary':
    base_value = 0.5
elif metric == 'logit':
    base_value = 0
else:
    raise ValueError('Invalid metric!')
results = [saved for saved in tf.io.gfile.listdir(cb_dir)
           if 'agg' not in saved and '.h5' in saved]
experiment_number = 0
if len(results):
    experiment_number += np.max([int(result.split('.')[-2].split('_')[-1][1:]) 
                            for result in results]) + 1
#print(experiment_number)
if 'arthur' in socket.gethostname():
    save_dir = os.path.join(
        cb_dir, '{}.h5'.format(socket.gethostname()[-1] + str(experiment_number).zfill(5))
    )
else:
    save_dir = os.path.join(
        cb_dir, '{}.h5'.format('0' + str(experiment_number).zfill(5))
    )
# mem_tmc = np.zeros((0, len(players)))
# idxs_tmc = np.zeros((0, len(players))).astype(int)
# with h5py.File(save_dir, 'w') as foo:
#     foo.create_dataset("mem_tmc", data=mem_tmc, compression='gzip')
#     foo.create_dataset("idxs_tmc", data=idxs_tmc, compression='gzip')
## Running CB-Shapley
c = None
if c is None:
    c = {i: np.array([i]) for i in range(len(players))}
elif not isinstance(c, dict):
    c = {i: np.where(np.array(c)==i)[0] for i in set(list(c))}

if tf.io.gfile.exists(os.path.join(cb_dir, 'chosen_players.txt')):
    chosen_players = open(os.path.join(
            cb_dir, 'chosen_players.txt')).read()
    try:
        chosen_players = np.array(chosen_players.split(',')).astype(int)
    except:
        #issue: sometime the read file suffers from conflict during reading and writing at the same time. That results in chosen file being read as 'zero' due to which error occurs. Solution: just read the file again to possibly avoid the file read/write conflict.
        while(len(chosen_players)<1):
            chosen_players = open(os.path.join(
            cb_dir, 'chosen_players.txt')).read()
        chosen_players = np.array(chosen_players.split(',')).astype(int)

    # if len(chosen_players) == 1:
    #     break
else:
    chosen_players = None

base_path = 'G:/CUB_200_2011/CUB_200_2011'
data_dir =base_path+'/train_test_split/train/'
data_dir_test =base_path+'/train_test_split/test/'
imgDataGen_official_split = ImageDataGenerator(preprocessing_function = preprocess_input)
actual_test_gen  = imgDataGen_official_split.flow_from_directory(data_dir_test,
                    target_size=(224, 224),
                    color_mode='rgb',
                    class_mode='categorical',
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    seed=None,
                    #subset='validation',
                    interpolation='nearest')    


orig_acc, new_acc =  eval_model(
    model=model,
    players=players,
    chosen_players=chosen_players,    c=c ,actual_test_gen=actual_test_gen)
    
