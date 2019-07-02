import matplotlib.pyplot as plt
import numpy as np

l1 = [3.90321,3.350312,3.018083,2.940548,2.828022]
l2 = [2.677788, 2.700625, 2.757513, 2.798777,2.803569]
l3 = [2.686909,2.69389,2.734523, 2.741081,2.794549]
l4 = [2.758241,2.759799,2.740125,2.755326, 2.797807]
l6 = [2.874975 ,2.840259,2.832756,2.803688,2.804356]
l8 = [2.965618,2.928709,2.831793,2.758091, 2.859768]

l = [l2,l3,l4,l6,l8]

l1v = [2.991784,2.615783 ,2.401232,2.155976,2.076034]
l2v = [1.944711,2.025991,1.913076,1.913076,1.961686]
l3v = [1.92725, 1.974833 ,1.910136,2.019707,2.045646]
l4v = [1.900537,1.922983,1.987281,2.044873,1.924689]
l6v = [2.021359,2.015078,1.999512,2.093196, 1.976582]
l8v = [2.081918,2.123515,2.081409,2.095469,2.131043]

lv = [l2v,l3v,l4v,l6v,l8v]

l = np.array(l)/100
lv = np.array(lv)/100

x = np.arange(len(l1))
xlable = [64,128,256,512,1024]

n = [2,3,4,6,8]

plt.figure(figsize=(12,6))

for i in range(len(l)):
    plt.subplot(122)
    plt.plot(x,lv[i],'.-')
    plt.ylabel('Validation Loss (MAE)') 
    plt.xlabel('Number of Unit')
    plt.legend( ['%d Layers'%n[k] for k in range(len(n))], loc='upper right')
    plt.xticks(x,xlable)
    
    plt.subplot(121)
    plt.plot(x,l[i],'.-')
    plt.ylabel('Training Loss (MAE)') 
    plt.xlabel('Number of Unit')
    plt.legend( ['%d Layers'%n[k] for k in range(len(n))], loc='upper right')
    plt.xticks(x,xlable)
    
plt.tight_layout()

#plt.ylim([0,.2])


plt.show()