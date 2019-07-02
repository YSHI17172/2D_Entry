from TwoD_Entry_bvp import Entry
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

entry_model = Entry()

samples = np.load("/work/yshi/two/sample_all_Train10000_200.npy")
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
ss = 5
step = 1/(10**ss)

model_name = 'Entry28_fold_2_repeating_0_2019-06-18_10:12:53.h5'
DNN_Model = load_model('model/'+model_name)

tf = 101.83174508
trajectory = []
contorl = []
hight = 50
counter = 0
while hight > 0 and counter < 2000/step:
    prediction = DNN_Model.predict(next_state.reshape((1,4)))
    
    alpha = prediction[0,0]
    
    anti_norm_sate = anti_normalize(next_state,r,v)
    state_dot = entry_model.EoM(anti_norm_sate,alpha) # t,state, tf
    # print (i,'a',alpha*180/np.pi)
    #print ('state', (anti_norm_sate[0] - entry_model.constant['R0'])/1000,anti_norm_sate[1]*entry_model.constant['R0']/1000,anti_norm_sate[2])
    # print ('dot',state_dot)
    
    next_state = anti_norm_sate.reshape((4,1)) + state_dot*step
    next_state = normalize(next_state.flatten(),r,v)
    
    trajectory.append(anti_norm_sate)
    contorl.append(alpha*180/np.pi)
    
    counter += 1 
    hight = anti_norm_sate[0] - entry_model.constant['R0']

trajectory = np.array(trajectory)

trajecotry = np.column_stack((trajectory,contorl))

np.save('trajectory/%s_%d'%(model_name,ss),trajecotry)
