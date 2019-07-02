import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt

#path_to_results = 'C:\\Users\\yshi\\OneDrive - University of South Carolina\\New\\Feature_Net\\graph_2D_CNN\\datasets\\results\\'

dataset = 'new_isolated_c20_cluster'

path_to_results = '/Users/User/Desktop/results/11/'
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

my_prec = 2 # desired precision

plot_loss = [[]] *6

batch = [32,64,128,256,512,1024]

fold = [6,4,1,0,1,2]

for name in results_names:
    time_stamp = re.findall(r'(.+)_results',name)[0]
    print ('=======',name,'=======')
    for pnm in parameter_names:
        if time_stamp in pnm:
            para_name=pnm
            with open(path_to_parameter + para_name, 'r') as my_file:
                parameter = json.load(my_file)
                        
                for key in parameter:
                    if key == 'batch_size':
                        size = parameter[key]
    with open(path_to_results + name, 'r') as my_file:
        tmp = json.load(my_file)
    histories = tmp['histories']
    
    index = batch.index(size)
    
    plot_loss[index] = histories[fold[index]]
         
    
    # print ('==================')
    # print ('val_loss')
    # val_loss = [np.amin(fold['val_loss'])*100 for fold in histories]
    # print ('mean:', round(np.mean(val_loss),my_prec))
    # print ('median:', round(np.median(val_loss),my_prec))
    # print ('max:', round(max(val_loss),my_prec))
    # print ('min:', round(min(val_loss),my_prec),' fold %d'%np.argmin(val_loss))
    # print ('stdev', round(np.std(val_loss),my_prec))
    # 
    
    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # 
    # for h in histories:
    #     #plt.plot(h['loss'],'.-')
    #     plt.plot(h['val_loss'],'.-')  
    # plt.legend(range(len(histories)), loc='upper right')
plt.figure(figsize=(6,5))

for b in plot_loss:
    # plt.subplot(122)
    # plt.plot(b['val_loss'])
    # plt.ylabel('Validation Loss') 
    # plt.xlabel('Epoch')
    # plt.legend(['batch size = %d'%b for b in batch], loc='upper right')
    
    #plt.subplot(121)
    plt.plot(b['loss'])
    plt.ylabel('Training Loss') 
    plt.xlabel('Epoch')
    plt.legend(['batch size = %d'%b for b in batch], loc='upper right')
    plt.tight_layout()

#plt.ylim([0,.2])


plt.show()
    
    
    


    