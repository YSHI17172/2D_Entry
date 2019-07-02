import matplotlib.pyplot as plt
import numpy as np
import os
import json

p1 = '/Users/User/Desktop/results/22/'
p2 = '/Users/User/Desktop/results/23/'

r1 = os.listdir(p1)
r2 = os.listdir(p2)

try:
    r1.remove('.DS_Store')
except:
    pass

try:
    r2.remove('.DS_Store')
except:
    pass


#remove folders
for f in r1[:]:
    if '.' not in f:
        r1.remove(f)
        
for f in r2[:]:
    if '.' not in f:
        r2.remove(f)     

for name in r1:
    with open(p1 + name, 'r') as my_file:
        tmp = json.load(my_file)
    h1 = tmp['histories'][0]

for name in r2:
    with open(p2 + name, 'r') as my_file:
        tmp = json.load(my_file)
    h2 = tmp['histories'][8]

plt.figure(figsize=(12,6))


plt.subplot(121)
plt.plot(h1['loss'])
plt.plot(h2['loss'])
plt.ylabel('Training Loss (MAE)') 
plt.xlabel('Epoch')
plt.ylim([0,.22])
plt.legend( ['Min-Max Scaling','Z-score'], loc='upper right')
#plt.xticks(x,xlable)

plt.subplot(122)
plt.plot(h1['val_loss'])
plt.plot(h2['val_loss'])
plt.ylabel('Validation Loss (MAE)') 
plt.xlabel('Epoch')
plt.legend( ['Min-Max Scaling','Z-score'], loc='upper right')
plt.ylim([0,.22])

#plt.legend( ['%d Layers'%n[k] for k in range(len(n))], loc='upper right')
#plt.xticks(x,xlable)
    
plt.tight_layout()

#plt.ylim([0,.2])


plt.show()



