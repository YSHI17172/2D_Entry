from TwoD_Entry_bvp import Entry
from keras.models import load_model
import numpy as np

entry_model = Entry()

hs = entry_model.constant['hs']
R0 = entry_model.constant['R0']
rho0 = entry_model.constant['rho0']
b0 = entry_model.constant['b0']
b2 = entry_model.constant['b2']
c1 = entry_model.constant['c1']
A = entry_model.constant['A']
m = entry_model.constant['m']
g0 = 9.81

def anti_normalize(state):
    r0 = state[0]
    v0 = state[2]
    
    v0 = v0*np.sqrt(R0*g0)
    r0 = r0*R0
    
    return np.array([r0,state[1],v0,state[-1]])


r = entry_model.constant['r0']
v = entry_model.constant['v0']
theta = 0
gamma = entry_model.constant['gamma0']
alpha = 0.3324101876204215

state = [r/R0,theta,v/np.sqrt(R0*g0),gamma]
state_dot = entry_model.EoM_normalized(state,alpha) # array

next_state = np.concatenate(state,state_dot)

#state = np.array(state).reshape((1,4)) 

mesh_size = 1011
ss = 1
step = 1/(10**ss)

model_path = '/Users/User/Desktop/code/models/30/'
model_name = 'Entry30_fold_1_repeating_2_2019-06-23_17:18:22.h5'
#DNN_Model = load_model('model/'+model_name)
DNN_Model = load_model(model_path+model_name)

tf = 101.83174508
trajectory = []
contorl = []
hight = 50
counter = 0
while hight > 0 and counter < 2000/step:
    trajectory.append(anti_normalize(next_state))
    
    prediction = DNN_Model.predict(next_state.reshape((1,8)))
    
    alpha = prediction[0,0]
    
    state_dot = entry_model.EoM_normalized(next_state[:4],alpha) # t,state, tf
    # print (i,'a',alpha*180/np.pi)
    #print ('state', (anti_norm_sate[0] - entry_model.constant['R0'])/1000,anti_norm_sate[1]*entry_model.constant['R0']/1000,anti_norm_sate[2])
    # print ('dot',state_dot)
    
    next_state_half = next_state[:4].reshape((4,1)) + state_dot*step
    next_state = np.concatenate(next_state_half,state_dot)
    
    contorl.append(alpha*180/np.pi)
    
    counter += 1 
    hight = anti_norm_sate[0] - entry_model.constant['R0']

trajectory = np.array(trajectory)

trajecotry = np.column_stack((trajectory,contorl))

np.save('trajectory/%s_%d'%(model_name,ss),trajecotry)
