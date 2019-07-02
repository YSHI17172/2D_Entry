import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt

path_to_results = '/Users/User/Desktop/results/'
path_to_parameter = '/Users/User/Desktop/results/parameters/'

results_names = os.listdir(path_to_results)
parameter_names = os.listdir(path_to_parameter)
try:
    results_names.remove('.DS_Store')
except:
    pass

#remove folders
for f in results_names[:]:
    if '.' not in f:
        results_names.remove(f)

my_prec = 6 # desired precision

for name in results_names:
    time_stamp = re.findall(r'(.+)_results',name)[0]
    print ('=======',name,'=======')
    for pnm in parameter_names:
        if time_stamp in pnm:
            para_name=pnm
            with open(path_to_parameter + para_name, 'r') as my_file:
                parameter = json.load(my_file)
                        
                for key in parameter:
                    print(key,':',parameter[key])
                print(para_name)
    with open(path_to_results + name, 'r') as my_file:
        tmp = json.load(my_file)
    histories = tmp['histories']
 
    keys = [key for key in histories[0]]
    print('keys:', keys)
    
    val_loss_min = []
    for h in histories:
        val_loss = [fold['val_loss'][-1]*100 for fold in histories]
        val_loss_min.append(np.amin(val_loss))
        
        min_val_loss = [min(fold['val_loss'])for fold in histories]
      
        max_idx = [histories[i]['val_loss'].index(min_val_loss[i]) for i in range(len(histories))] 
            
    #vals = [elt[1] for elt in tmp['outputs']]         
            
        
    print ('==================')
    print ('Loss')
    train_loss = [np.amin(fold['loss'])*100 for fold in histories]
    print ('mean:', round(np.mean(train_loss),my_prec))
    print ('median:', round(np.median(train_loss),my_prec))
    print ('max:', round(max(train_loss),my_prec))
    print ('min:', round(min(train_loss),my_prec),' fold %d'%np.argmin(train_loss))
    print ('stdev', round(np.std(train_loss),my_prec))
   
    print ('==================')
    print ('mean_absolute_error')
    mean_absolute_error = [np.amin(fold['mean_absolute_error'])*100 for fold in histories]
    print ('mean:', round(np.mean(mean_absolute_error),my_prec))
    print ('median:', round(np.median(mean_absolute_error),my_prec))
    print ('max:', round(max(mean_absolute_error),my_prec))
    print ('min:', round(min(mean_absolute_error),my_prec),' fold %d'%np.argmin(mean_absolute_error))
    print ('stdev', round(np.std(mean_absolute_error),my_prec))
    
    if 'mean_squared_error' in keys:
        print ('==================')
        print ('mean_squared_error')
        mean_squared_error = [np.amin(fold['mean_squared_error'])*100 for fold in histories]
        print ('mean:', round(np.mean(mean_squared_error),my_prec))
        print ('median:', round(np.median(mean_squared_error),my_prec))
        print ('max:', round(max(mean_squared_error),my_prec))
        print ('min:', round(min(mean_squared_error),my_prec),' fold %d'%np.argmin(mean_squared_error))
        print ('stdev', round(np.std(mean_squared_error),my_prec))
        
    
    print ('==================')
    print ('val_loss')
    val_loss = [np.amin(fold['val_loss'])*100 for fold in histories]
    print ('mean:', round(np.mean(val_loss),my_prec))
    print ('median:', round(np.median(val_loss),my_prec))
    print ('max:', round(max(val_loss),my_prec))
    print ('min:', round(min(val_loss),my_prec),' fold %d'%np.argmin(val_loss))
    print ('stdev', round(np.std(val_loss),my_prec))
    
    print ('==================')
    print ('val_mean_absolute_error')
    val_mean_absolute_error = [np.amin(fold['val_mean_absolute_error'])*100 for fold in histories]
    print ('mean:', round(np.mean(val_mean_absolute_error),my_prec))
    print ('median:', round(np.median(val_mean_absolute_error),my_prec))
    print ('max:', round(max(val_mean_absolute_error),my_prec))
    print ('min:', round(min(val_mean_absolute_error),my_prec),' fold %d'%np.argmin(val_mean_absolute_error))
    print ('stdev', round(np.std(val_mean_absolute_error),my_prec))
    
    if 'mean_squared_error' in keys:
        print ('==================')
        print ('val_mean_squared_error')
        val_mean_squared_error = [np.amin(fold['val_mean_squared_error'])*100 for fold in histories]
        print ('mean:', round(np.mean(val_mean_squared_error),my_prec))
        print ('median:', round(np.median(val_mean_squared_error),my_prec))
        print ('max:', round(max(val_mean_squared_error),my_prec))
        print ('min:', round(min(val_mean_squared_error),my_prec),' fold %d'%np.argmin(val_mean_squared_error))
        print ('stdev', round(np.std(val_mean_squared_error),my_prec))

#    for h in histories:
#        plt.figure()
#        
#        #if 'mean_squared_error' in keys:
#        #plt.subplot(121)
#        plt.plot(h['loss'],'.-')
#        plt.plot(h['val_loss'],'.-')
#        plt.xlabel('Epoch')
#        plt.ylabel('Loss')
#        plt.legend(['Train', 'Test'], loc='upper right')
        #plt.ylim([0,0.1])
#         
#         # if 'mean_squared_error' in keys:
#         #     #plt.subplot(122)
#         #     plt.plot(h['mean_squared_error'],'.-')
#         #     plt.plot(h['val_mean_squared_error'],'.-')
#         #     plt.ylabel('Mean Squared Error')
#         #     plt.xlabel('Epoch')
#         #     plt.ylim([0,.2])
#         #     plt.legend(['Train', 'Test'], loc='upper right')
#         #     
#         # if 'mean_absolute_error' in keys:
#         #     plt.subplot(122)
#         #     plt.plot(h['mean_absolute_error'],'.-')
#         #     plt.plot(h['val_mean_absolute_error'],'.-')
#         #     plt.ylabel('Mean Absolute Error')
#         #     plt.xlabel('Epoch')
#         #     #plt.ylim([0,.2])
#         #     plt.legend(['Train', 'Test'], loc='upper right')
#         #     
plt.show()
    
    
    


    