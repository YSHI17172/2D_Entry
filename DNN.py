#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:51:35 2019

@author: Yang Shi
"""
from TwoD_Entry_bvp import Entry

import argparse

import math
import numpy as np
import os
import sys
import random
import json
import datetime
import time

from keras.layers import Dense, Input
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping

parser = argparse.ArgumentParser()

# optional arguments
parser.add_argument('--n_folds', type=int, default=5, choices=[2,3,4,5,6,7,8,9,10], help='number of folds for cross-validation')
parser.add_argument('--n_repeats', type=int, default=3, choices=[1,2,3,4,5], help='number of times each fold should be repeated')
parser.add_argument('--batch_size', type=int, default=32, choices=[16,32,64,128,256,512,1024], help='batch size')
parser.add_argument('--nb_epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=10, help='patience for early stopping strategy')
parser.add_argument('--drop_rate', type=float, default=0.3, help='dropout rate')
parser.add_argument('--dense_units', type=int, default=256, choices=[16,32,64,128,256,512],help='number of units of hidden layer')

args = parser.parse_args()

# convert command line arguments

n_folds = args.n_folds
dense_units = args.dense_units
n_repeats = args.n_repeats
batch_size = args.batch_size
nb_epochs = args.nb_epochs
my_patience = args.patience
drop_rate = args.drop_rate
dense_units = args.dense_units

my_optimizer = 'adam'
my_loss_function = 'mae'
activation_hidden = 'relu'

# =============================================================================

def main():
    
    my_date_time = '_'.join(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").split())

    parameters = {
                  'n_folds':n_folds,
                  'n_repeats':n_repeats,
                  'batch_size':batch_size,
                  'nb_epochs':nb_epochs,
                  'my_patience':my_patience,
                  'drop_rate':drop_rate,
                  'my_optimizer':my_optimizer
                  }
    
    path_root = sys.path[0]

    name_save = path_root + '/results/' + my_date_time
      
    print ('========== loading samples ==========')
    
    samples = np.load("sample_all_Train10000_200.npy")
    entry_model = Entry()
    tf = samples[:,0,-1]
    altitute = (samples[:,1,0] - entry_model.constant['R0'])/1000
    downrange = samples[:,2,-1]*entry_model.constant['R0']/1000 - 100
    velocity = samples[:,3,0]
    gamma = samples[:,4,0]*180/np.pi
    
    print ('Tf range: max:%.2f,min:%.2f, [s],'%(np.amax(tf),np.amin(tf)))
    print ('Downrange range: max:%.2f,min:%.2f [km],'%(np.amax(downrange),np.amin(downrange)))
    print ('Initial Altitute range: max:%.2f,min:%.2f,[km]'%(np.amax(altitute),np.amin(altitute)))
    print ('Initial Velocity range:max:%d,min:%d, [m/s]'%(np.amax(velocity),np.amin(velocity)))
    print ('Initial Flight Path Angle range: max:%.2f,min:%.2f,'%(np.amax(gamma),np.amin(gamma)))

    alpha = samples[:,-1,:]

    ys = alpha.flatten() #output row style
    
    r = (samples[:,1,:].flatten() - entry_model.constant['R0'])/entry_model.constant['h0']
    theta = samples[:,2,:].flatten()
    v = samples[:,3,:].flatten() / entry_model.constant['v0']
    g = samples[:,4,:].flatten()

    tensors = np.column_stack((r,theta,v,g))
    tensors = tensors.astype(np.float32)

    print ('input shape:', tensors.shape)

    print ('========== shuffling data ==========')

    shuffled_idxs = random.sample(range(tensors.shape[0]), int(tensors.shape[0])) # sample w/o replct
    tensors = tensors[shuffled_idxs]
    ys = ys[shuffled_idxs]
    shuffled_idxs = np.array(shuffled_idxs)
    
    print ('========== conducting', n_folds ,'fold cross validation =========='); 
    print ('repeating each fold:', n_repeats, 'times')

    folds = np.array_split(tensors,n_folds,axis=0)

    print ('fold sizes:', [fold.shape[0] for fold in folds])

    folds_labels = np.array_split(ys,n_folds,axis=0)
    
    outputs = []
    histories = []

    for i in range(n_folds):
        
        t = time.time()
        
        x_train = np.concatenate([fold for j,fold in enumerate(folds) if j!=i],axis=0)
        x_test = [fold for j,fold in enumerate(folds) if j==i]
        
        y_train = np.concatenate([y for j,y in enumerate(folds_labels) if j!=i],axis=0)
        y_test = [y for j,y in enumerate(folds_labels) if j==i]
        
        for repeating in range(n_repeats):
            
            print ('clearing Keras session')
            K.clear_session()
            
            my_input = Input(shape=(4,), dtype='float32')
            
            dense_1 = Dense(dense_units,
                            activation=activation_hidden)(my_input)
            
            dense_2 = Dense(dense_units,
                            activation=activation_hidden)(dense_1)
            
            dense_3 = Dense(dense_units,
                            activation=activation_hidden)(dense_2)
            
            dense_4 = Dense(dense_units,
                            activation=activation_hidden)(dense_3)
                            
            dense_5 = Dense(dense_units,
                            activation=activation_hidden)(dense_4)
                            
            dense_6 = Dense(dense_units,
                            activation=activation_hidden)(dense_5)
            
            prob = Dense(1)(dense_6)
        
            # instantiate model
            model = Model(my_input,prob)
                            
            # configure model for training
            model.compile(loss=my_loss_function,
                          optimizer=my_optimizer,
                          metrics=['mae','mse'])
            
            print ('model compiled')
            
           # early_stopping = EarlyStopping(monitor='val_mean_absolute_error', # go through epochs as long as acc on validation set increases
           #                                patience=my_patience,
           #                                mode='max') 
            
            history = model.fit(x_train,
                                y_train,
                                batch_size=batch_size,
                                nb_epoch=nb_epochs,
                                validation_data=(x_test, y_test),)
                               # callbacks=[early_stopping])

            # save [min loss,max acc] on test set
            max_acc = max(model.history.history['val_mean_absolute_error'])
            max_idx = model.history.history['val_mean_absolute_error'].index(max_acc)
            output = [model.history.history['val_loss'][max_idx],max_acc]
            outputs.append(output)
            
            # also save full history for sanity checking
            histories.append(model.history.history)
                   
        print ('**** fold', i+1 ,'done in ' + str(math.ceil(time.time() - t)) + ' second(s) ****')

    # save results to disk
    with open(name_save + '_parameters.json', 'w') as my_file:
        json.dump(parameters, my_file, sort_keys=True, indent=4)

    print ('========== parameters defined and saved to disk ==========')

    with open(name_save + '_results.json', 'w') as my_file:
        json.dump({'outputs':outputs,'histories':histories}, my_file, sort_keys=False, indent=4)

    print( '========== results saved to disk ==========')

if __name__ == "__main__":
    main()

