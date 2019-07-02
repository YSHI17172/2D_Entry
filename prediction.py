from keras.models import load_model
import numpy as np
from keras import backend as K
import os

model_path = '/home/yshi/two/model/28/'

train = np.load("/work/yshi/two/sample_all_Train10000_200.npy")

samples = np.load('/work/yshi/two/sample_all_Train.npy')

r0 = train[:,1,:].flatten()
v0 = train[:,3,:].flatten()

alpha = samples[:,-1,:]*np.pi/180

theta = samples[:,2,:].flatten()
g = samples[:,4,:].flatten()

r = samples[:,1,:].flatten()
r = (r-np.mean(r0))/np.std(r0)

v = samples[:,3,:].flatten()
v = (v-np.mean(v0))/np.std(v0)
tensors = np.column_stack((r,theta,v,g))
tensors = tensors.astype(np.float32)


DNN_Model = load_model(model_path+'Entry26_fold_3_repeating_1_2019-06-11_17:07:21.h5')
predictions = DNN_Model.predict(tensors)

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
    comp.append(compare)
    print(np.mean(np.abs(compare)))


#compare = predictions - alpha.flatten()

