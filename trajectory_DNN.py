#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:17:36 2019

@author: Yang Shi
"""

from TwoD_Entry_bvp import Entry
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

entry_model = Entry()

samples = np.load("sample_all_Train10000_200.npy")
r = samples[:,1,:].flatten()
v = samples[:,3,:].flatten()

def normalize(state,r,v):  
    r0 = state[0]
    v0 = state[2]
    v0 = (v0-np.mean(v))/np.std(v)
    r0 = (r0-np.mean(r))/np.std(r)

    return np.array([r0,state[1],v0,state[-1]])

def anti_normalize(state,r,v):
    r0 = state[0]
    v0 = state[2]
    
    v0 = v0*np.std(v) + np.mean(v) 
    r0 = r0*np.std(r) + np.mean(r)
    
    return np.array([r0,state[1],v0,state[-1]])

state = [entry_model.constant['r0'],0,entry_model.constant['v0'],entry_model.constant['gamma0']]  #r,theta,v,gamma
next_state = normalize(state,r,v)

#state = np.array(state).reshape((1,4))

mesh_size = 1011

model_path = '/Users/User/Desktop/code/models/'
DNN_Model = load_model(model_path+'Entry28_fold_2_repeating_1_2019-06-18_10:12:53.h5')

tf = 101.83174508
trajectory = []
contorl = []
for i in range(mesh_size):
    prediction = DNN_Model.predict(next_state.reshape((1,4)))
    
    alpha = prediction[0,0]
    
    anti_norm_sate = anti_normalize(next_state,r,v)
    state_dot = entry_model.EoM(anti_norm_sate,alpha) # t,state, tf
    # print (i,'a',alpha*180/np.pi)
    # print ('state', (anti_norm_sate[0] - entry_model.constant['R0'])/1000,anti_norm_sate[1]*entry_model.constant['R0']/1000 - 100,anti_norm_sate[2])
    # print ('dot',state_dot)
    
    next_state = anti_norm_sate.reshape((4,1)) + state_dot*tf/mesh_size
    next_state = normalize(next_state.flatten(),r,v)
    
    trajectory.append(anti_norm_sate)
    contorl.append(alpha*180/np.pi)

trajectory = np.array(trajectory)

#indirect method results
sol = entry_model.get_optimize([14]) #14 initial guess of tf for 0 downrange


plt.subplot(221)
plt.plot(trajectory[:,1]*entry_model.constant['R0']/1000,(trajectory[:,0]-entry_model.constant['R0'])/1000,label = 'DNN-based Approach')
plt.plot(sol.y[1]*entry_model.constant['R0']/1000,(sol.y[0]-entry_model.constant['R0'])/1000,label='Indirect Method')
plt.xlabel('Downrange [km]',)
plt.ylabel('Altitude [km]',)
plt.legend()
#plt.legend(['batch size = %d'%b for b in batch], loc='upper right')

plt.subplot(222)
plt.plot(sol.x*sol.p,trajectory[:,2],label = 'DNN-based Approach')
plt.plot(sol.x*sol.p,sol.y[2],label='Indirect Method')
plt.xlabel('Time [s]',);
plt.ylabel('Velocity [m/s]',);
plt.legend()

plt.subplot(223)
plt.plot(sol.x*sol.p,trajectory[:,3],label = 'DNN-based Approach')
plt.plot(sol.x*sol.p,sol.y[3],label='Indirect Method')
plt.xlabel('Time [s]',);
plt.ylabel('Flight Path Angle [deg]',);
plt.legend()

plt.subplot(224)
plt.plot(sol.x*sol.p,contorl,label = 'DNN-based Approach')
plt.plot(sol.x*sol.p,entry_model.constant['c1']*sol.y[-1,:]*180/(2*entry_model.constant['b2']*sol.y[2,:]*sol.y[6,:]*np.pi),label='Indirect Method')
plt.xlabel('Time [s]',);
plt.ylabel('Optimal Control alpha [deg]',)
plt.legend()

plt.figure(figsize=(8,6))

plt.show()