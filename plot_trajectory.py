import numpy as np
import matplotlib.pyplot as plt
from TwoD_Entry_bvp import Entry

entry_model = Entry()

data = np.load('/Users/User/Desktop/code/trajectory/Entry30_fold_1_repeating_2_0.npy')

contorl = data[:,-1]

trajectory = data[:,:-1]

step = 1e-4

time = [i*step for i in range(len(contorl))]

#indirect method results
sol = entry_model.get_optimize([14]) #14 initial guess of tf for 0 downrange

plt.figure(figsize=(14,7))

plt.subplot(221)
plt.plot(trajectory[:,1]*entry_model.constant['R0']/1000,(trajectory[:,0]-entry_model.constant['R0'])/1000,label = 'DNN-based Approach')
plt.plot(sol.y[1]*entry_model.constant['R0']/1000,(sol.y[0]-entry_model.constant['R0'])/1000,label='Indirect Method')
plt.xlabel('Downrange [km]',)
plt.ylabel('Altitude [km]',)
plt.legend()
#plt.legend(['batch size = %d'%b for b in batch], loc='upper right')

plt.subplot(222)
plt.plot(time,trajectory[:,2],label = 'DNN-based Approach')
plt.plot(sol.x*sol.p,sol.y[2],label='Indirect Method')
plt.xlabel('Time [s]',);
plt.ylabel('Velocity [m/s]',);
plt.legend()

plt.subplot(223)
plt.plot(time,trajectory[:,3],label = 'DNN-based Approach')
plt.plot(sol.x*sol.p,sol.y[3],label='Indirect Method')
plt.xlabel('Time [s]',);
plt.ylabel('Flight Path Angle [deg]',);
plt.legend()

plt.subplot(224)
plt.plot(time,contorl,label = 'DNN-based Approach')
plt.plot(sol.x*sol.p,entry_model.constant['c1']*sol.y[-1,:]*180/(2*entry_model.constant['b2']*sol.y[2,:]*sol.y[6,:]*np.pi),label='Indirect Method')
plt.xlabel('Time [s]',);
plt.ylabel('Optimal Control alpha [deg]',)
plt.legend()


plt.show()