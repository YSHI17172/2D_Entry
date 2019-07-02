import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt

path_to_results = '/Users/User/Desktop/results/19/'
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

plot_loss = [[]] *7

lr = [0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]

fold = [0,0,3,8,0,3,9]

for name in results_names:
    time_stamp = re.findall(r'(.+)_results',name)[0]
    print ('=======',name,'=======')
    for pnm in parameter_names:
        if time_stamp in pnm:
            para_name=pnm
            with open(path_to_parameter + para_name, 'r') as my_file:
                parameter = json.load(my_file)
                        
                for key in parameter:
                    if key == 'learning_rate':
                        size = parameter[key]
    with open(path_to_results + name, 'r') as my_file:
        tmp = json.load(my_file)
    histories = tmp['histories']
    
    index = lr.index(size)
    
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
plt.figure(figsize=(12,6))

for b in plot_loss:
    plt.subplot(122)
    plt.plot(b['val_loss'])
    plt.ylabel('Validation Loss (MAE)') 
    plt.xlabel('Epoch')
    plt.legend(['Learning Rate = %g'%b for b in lr], loc='upper right')
    plt.ylim([0,.2])
    
    plt.subplot(121)
    plt.plot(b['loss'])
    plt.ylabel('Training Loss (MAE)') 
    plt.xlabel('Epoch')
    plt.legend(['Learning Rate = %g'%b for b in lr], loc='upper right')
    plt.ylim([0,.2])


plt.tight_layout()

#plt.ylim([0,.2])
plt.show()
    
    
    


    