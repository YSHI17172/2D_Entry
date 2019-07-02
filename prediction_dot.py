from keras.models import load_model
import numpy as np
from keras import backend as K
import os
from TwoD_Entry_bvp import Entry

model_path = '/home/yshi/two/model/43/'

train = np.load("/work/yshi/two/sample_all_Train10000_200.npy")

samples = np.load('/work/yshi/two/sample_all_Train.npy')

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

alpha = samples[:,-1,:]*np.pi/180

theta = samples[:,2,:].flatten()
gamma = samples[:,4,:].flatten()

r = samples[:,1,:].flatten()
r_normalized = r/R0
v = samples[:,3,:].flatten()
v_normalized = v/np.sqrt(R0*g0)

rho = rho0*np.exp(-(r-R0)/hs)
Cl = c1*alpha.flatten()
Cd = b2*alpha.flatten()**2+b0
L = R0*rho*v_normalized**2*A*Cl/(2*m)
D = R0*rho*v_normalized**2*A*Cd/(2*m)

# ODE
r_dot     = v_normalized*np.sin(gamma)
theta_dot = v_normalized*np.cos(gamma)/r_normalized
v_dot     = -D - (np.sin(gamma)/r_normalized**2)
gamma_dot = L/v_normalized + (v_normalized*np.cos(gamma)/r_normalized) - (np.cos(gamma)/(v_normalized*r_normalized**2))

tensors = np.column_stack((r_normalized,theta,v_normalized,gamma,r_dot,theta_dot,v_dot,gamma_dot))
tensors = tensors.astype(np.float32)

#DNN_Model = load_model(model_path+'Entry26_fold_3_repeating_1_2019-06-11_17:07:21.h5')
#predictions = DNN_Model.predict(tensors)

predicts = []
model_name = os.listdir(model_path)
for mn in model_name:
    K.clear_session()
    DNN_Model = load_model(model_path+mn)
    predictions = DNN_Model.predict(tensors)
    predicts.append(predictions)
    np.save('predictions/prediction_%s'%mn,predictions)

comp = []
for pre in predicts:
    compare = pre.flatten() - alpha.flatten()
    comp.append(np.mean(np.abs(compare)))
    print(np.mean(np.abs(compare)))
print (np.mean(comp))
print (np.amax(comp))
print (np.amin(comp))
print (np.argmin(comp))
print (model_name[np.argmin(comp)])

#compare = predictions - alpha.flatten()

