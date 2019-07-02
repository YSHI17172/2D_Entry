#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:10:48 2019

@author: tliu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

class Entry:

    def  __init__(self, random = False):
        self.t = None
        self.state = None
        self.random = random
        self.thetaf = 0
        self.constant = {'hs' : 7200,
                         'rho0' : 1.225,
                         'R0' : 6.378137e6,
                         'mu' : 3.986004418e14,
                         'b0' : 0.0612, 'b2' : 1.6537, # CD =1.6537a2 +0.0612
                         'c1' : 1.5658, # CL =1.5658a
                         'm' : 100, # mass
                         'A' : 0.3, # A_ref
                         'h0' : 50000, # initial height, [m]
                         'theta0' : 0*np.pi/180, # initial downrange angle, [rad]
                         'v0' : 4000, # initial velocity, [m/s]
                         'gamma0' : -90*np.pi/180, # initial FPA, [rad]
                         'hf' : 0, # terminal height, [m]
                         }
        self.constant['r0'] = self.constant['h0'] + self.constant['R0'] # initial radial distance, [m]
        self.constant['rf'] = self.constant['hf'] + self.constant['R0'] # terminal radial distance, [m]
        self.constant['thetaf'] = 100000/self.constant['R0'] # terminal downrange angle, [rad]
        self.reset()
        
    def reset(self,state=None):
        if state == None:
            if self.random:
                np.random.seed() # reset random seed
                self.constant['h0'] =  np.random.randint(5000+1)+47500
                self.constant['r0'] = self.constant['R0'] + self.constant['h0']
                self.constant['v0'] = 3800 + np.random.randint(4000+1)/10
                self.constant['gamma0'] = (-90+np.random.randint(1000+1)*5/1000)*np.pi/180
                self.constant['thetaf'] = (97500+ np.random.randint(5000+1))/self.constant['R0'] # terminal downrange angle, [rad]
        else:
            self.constant['h0'] = state[0]
            self.constant['r0'] = self.constant['R0'] + self.constant['h0']
            self.constant['gamma0'] = state[3]
            self.constant['v0'] = state[2]
            self.constant['thetaf'] = state[1]
            
        self.state = np.array([self.constant['r0'],self.constant['theta0'],self.constant['v0'],self.constant['gamma0']])
  
    def get_sample(self,solution,new_mesh_size):
        mesh_size = solution.x.size
        tf = solution.p
        
        new_index = resize(mesh_size,200)  
        new_tao = np.linspace(0, 1, new_mesh_size)     
        #new_tao = solution.x[new_index]       
        new_guess = solution.y[:,new_index]       
        new_solution = solve_bvp(self.EoM_bvp,self.boundary_condition,new_tao,new_guess,tf,max_nodes=75000)
        
        alpha = self.constant['c1']*new_solution.y[-1,:]*180/(2*self.constant['b2']*new_solution.y[2,:]*new_solution.y[6,:]*np.pi)
        
        new_mesh = new_solution.x.flatten(order='F')
        new_mesh[-1] = new_mesh[-1]*new_solution.p
        
        sample = np.vstack((new_mesh,new_solution.y,alpha.flatten(order='F')))
        
        return sample
    
    def get_optimize(self,tf):
        alpha = 0;
        Cd = self.constant['b2']*alpha**2+self.constant['b0'];
        B = self.constant['m']/(Cd*self.constant['A']);
        C = self.constant['rho0']*self.constant['hs']/(2*B*np.sin(self.constant['gamma0']));
         
        t = np.linspace(0, 1, 1001)
        r_ae = np.linspace(self.constant['r0'],self.constant['R0'],t.size)
        v_ae = self.constant['v0']*np.exp(C*np.exp(-(r_ae-self.constant['R0'])/self.constant['hs']));
        lambda_v_ae = -2*v_ae
        
        init_guess = np.zeros((8,t.size))
        init_guess[0,:] = r_ae
        init_guess[2,:] = v_ae
        init_guess[3,:] = self.constant['gamma0']
        init_guess[6,:] = lambda_v_ae
        
        # a homotopy approach
        max_step = 21     
        while max_step < 102:
            continue_guess = init_guess
            tf_guess = tf
            tao = t
            downrange_dangle = np.linspace(0,self.constant['thetaf'],max_step)
            for i in range(max_step):
                self.thetaf = downrange_dangle[i]
                solution = solve_bvp(self.EoM_bvp,self.boundary_condition,tao,continue_guess,tf_guess,max_nodes=75000)
                
                #update initial guesses
                tao = np.linspace(0, 1, solution.x.size)
                continue_guess = solution.y
                tf_guess = solution.p
                
                # print('Step %d/%d Success: '%(i,max_step), solution.success)
                # if not solution.success:
                #     print(solution.message)
                # print(solution.x.shape,solution.y.shape)
                # #print(solution)
                # print('')
                if not solution.success:
                    break
            else:
                print('Steps %d Sucess'%max_step)
                return solution
            print('Step %d failed, start again'%max_step)
            max_step += 10

    def boundary_condition(self,X0,Xf,tf):
        
        hs = self.constant['hs']
        R0 = self.constant['R0']
        rho0 = self.constant['rho0']
        mu = self.constant['mu']
        b0 = self.constant['b0']
        b2 = self.constant['b2']
        c1 = self.constant['c1']
        A = self.constant['A']
        m = self.constant['m']
        
        r_tf,theta_tf,v_tf,gamma_tf,lambda_rf,lambda_tf,lambda_vf,lambda_gf  = Xf
        
        rho = rho0*np.exp(-(r_tf-R0)/hs)
        alphaf = c1*lambda_gf/(2*b2*v_tf*lambda_vf)
        
        Cl = c1*alphaf
        Cd = b2*alphaf**2+b0
        L = rho*v_tf**2*A*Cl/(2*m)
        D = rho*v_tf**2*A*Cd/(2*m)
        
        rf_dot = v_tf*np.sin(gamma_tf)
        thetaf_dot = v_tf*np.cos(gamma_tf)/r_tf
        vf_dot = -D - (mu*np.sin(gamma_tf)/r_tf**2)
        gammaf_dot = L/v_tf + v_tf*np.cos(gamma_tf)/r_tf-mu*np.cos(gamma_tf)/(v_tf*r_tf**2);

        #boundary conditions
        H_end = lambda_rf*rf_dot + lambda_tf*thetaf_dot + lambda_vf*vf_dot + lambda_gf*gammaf_dot
        rf_error = r_tf - self.constant['rf']
        thetaf_error = theta_tf - self.thetaf
        lambda_v_error = lambda_vf+2*v_tf
        
        r0_error = X0[0] - self.constant['r0']
        theta0_error = X0[1] - self.constant['theta0']
        v0_error = X0[2] - self.constant['v0'] 
        gamma0_error = X0[3] - self.constant['gamma0']
        
        bc = [r0_error,theta0_error,v0_error,gamma0_error,
              rf_error, thetaf_error, lambda_v_error, lambda_gf, H_end]
        
#        print (bc)
#        print ('')
        return np.array(bc)

    def EoM(self,X,alpha):
            
        r,theta,v,gamma = X
        
        hs = self.constant['hs']
        R0 = self.constant['R0']
        rho0 = self.constant['rho0']
        mu = self.constant['mu']
        b0 = self.constant['b0']
        b2 = self.constant['b2']
        c1 = self.constant['c1']
        A = self.constant['A']
        m = self.constant['m']
        
        rho = rho0*np.exp(-(r-R0)/hs)
        Cl = c1*alpha
        Cd = b2*alpha**2+b0
        L = rho*v**2*A*Cl/(2*m)
        D = rho*v**2*A*Cd/(2*m)
        
        # ODE
        r_dot     = v*np.sin(gamma)
        theta_dot = v*np.cos(gamma)/r
        v_dot     = -D - (mu*np.sin(gamma)/r**2)
        gamma_dot = L/v + (v*np.cos(gamma)/r) - (mu*np.cos(gamma)/(v*r**2)) 
    
        state_dot = np.vstack([r_dot, theta_dot, v_dot, gamma_dot])
        #print (state_dot[:,-1])
        return state_dot
        
    def EoM_normalized(self,X,alpha):
            
        r,theta,v,gamma = X
        
        hs = self.constant['hs']
        R0 = self.constant['R0']
        rho0 = self.constant['rho0']
        #mu = self.constant['mu']
        b0 = self.constant['b0']
        b2 = self.constant['b2']
        c1 = self.constant['c1']
        A = self.constant['A']
        m = self.constant['m']
        
        g0 = 9.81
        
        h = r-self.constant['R0']
        v_normalized = v/np.sqrt(R0*g0)
        r_normalized = r/R0
        
        rho = rho0*np.exp(-h/hs)
        Cl = c1*alpha
        Cd = b2*alpha**2+b0
        L = R0*rho*v_normalized**2*A*Cl/(2*m)
        D = R0*rho*v_normalized**2*A*Cd/(2*m)
        
        # ODE
        r_dot     = v_normalized*np.sin(gamma)
        theta_dot = v_normalized*np.cos(gamma)/r_normalized
        v_dot     = -D - (np.sin(gamma)/r_normalized**2)
        gamma_dot = L/v_normalized + (v_normalized*np.cos(gamma)/r_normalized) - (np.cos(gamma)/(v_normalized*r_normalized**2)) 
    
        state_dot = np.vstack([r_dot*R0, theta_dot, v_dot*np.sqrt(R0*g0), gamma_dot])
        #print (state_dot[:,-1])
        return state_dot/np.sqrt(R0/g0)
   
    def EoM_bvp(self,t,X,tf):
        
        r,theta,v,gamma,lam_r,lam_t,lam_v,lam_g = X
        
        hs = self.constant['hs']
        R0 = self.constant['R0']
        rho0 = self.constant['rho0']
        mu = self.constant['mu']
        b0 = self.constant['b0']
        b2 = self.constant['b2']
        c1 = self.constant['c1']
        A = self.constant['A']
        m = self.constant['m']
        
        rho = rho0*np.exp(-(r-R0)/hs)
        
        alpha = c1*lam_g/(2*b2*v*lam_v)
        Cl = c1*alpha
        Cd = b2*alpha**2+b0
        L = rho*v**2*A*Cl/(2*m)
        D = rho*v**2*A*Cd/(2*m)
        Dr = -A*Cd*rho*v**2/(2*m*hs)
        Dv = A*Cd*rho*v/m
        Lr = -A*Cl*rho*v**2/(2*m*hs)
        Lv = A*Cl*rho*v/m
        
        f13 = np.sin(gamma)
        f14 = v*np.cos(gamma)
        f21 = -v*np.cos(gamma)/r**2
        f23 = np.cos(gamma)/r
        f24 = -v*np.sin(gamma)/r
        f31 = -Dr + (2*mu*np.sin(gamma)/r**3)
        f33 = -Dv
        f34 = -mu*np.cos(gamma)/r**2
        f41 = Lr/v-v*np.cos(gamma)/r**2+2*mu*np.cos(gamma)/(v*r**3)
        f43 = Lv/v-L/v**2+np.cos(gamma)/r+mu*np.cos(gamma)/(v**2*r**2)
        f44 = -v*np.sin(gamma)/r+mu*np.sin(gamma)/(v*r**2)
        
        # ODE
        r_dot     = v*np.sin(gamma)
        theta_dot = v*np.cos(gamma)/r
        v_dot     = -D - (mu*np.sin(gamma)/r**2)
        gamma_dot = L/v + (v*np.cos(gamma)/r) - (mu*np.cos(gamma)/(v*r**2)) 
        
        lam_r_dot = -(f21*lam_t + f31*lam_v + f41*lam_g)
        lam_t_dot = np.zeros_like(r_dot)
        lam_v_dot = -(f13*lam_r + f23*lam_t + f33*lam_v + f43*lam_g)
        lam_g_dot = -(f14*lam_r + f24*lam_t + f34*lam_v + f44*lam_g)

        state_dot = np.vstack([r_dot, theta_dot, v_dot, gamma_dot, lam_r_dot, lam_t_dot, lam_v_dot, lam_g_dot])
        #print (state_dot[:,-1])
        return state_dot*tf
    
def resize(m,n):
    ### resize m length list to n length list with equal spacing
    
    if n >= m : 
        return np.arange(m)
    else:
        o = np.arange(m)
        one = m//n
        if one > 1:
            new =  o[::one]
        else:
            new = o      
        while new.size > n:
            two = new.size//(new.size-n) + 1
            remove = new[::two]
            new = np.setdiff1d(new,remove)      
        return new
    
if __name__ == '__main__':
    from matplotlib.ticker import FormatStrFormatter
    test = Entry()
    res = test.get_optimize([14])
    #s = test.get_sample(res)
    
    plt.figure(figsize=(8,6))
    res = [res]
    
    for sol in res:
        plt.subplot(221)
        plt.plot(sol.y[1]*test.constant['R0']/1000,(sol.y[0]-test.constant['R0'])/1000)
        plt.xlabel('Downrange [km]',)
        plt.ylabel('Altitude [km]',)
        
        plt.subplot(222)
        plt.plot(sol.x*sol.p,sol.y[2])
        plt.xlabel('Time [s]',);
        plt.ylabel('Velocity [m/s]',);
        
        plt.subplot(223)
        plt.plot(sol.x*sol.p,sol.y[3])
        plt.xlabel('Time [s]',);
        plt.ylabel('Flight Path Angle [deg]',);
        
        plt.subplot(224)
        plt.plot(sol.x*sol.p,test.constant['c1']*sol.y[-1,:]*180/(2*test.constant['b2']*sol.y[2,:]*sol.y[6,:]*np.pi))
        plt.xlabel('Time [s]',);
        plt.ylabel('Optimal Control alpha [deg]',);
        
    plt.figure(figsize=(8,6))
    
    for sol in res:
        plt.subplot(221)
        plt.plot(sol.x*sol.p,sol.y[4])
        plt.xlabel('Time [s]',)
        plt.ylabel(r'$\lambda_{r}$ $[m/s^{2}]$')
        
        plt.subplot(222)
        plt.plot(sol.x*sol.p,sol.y[5])
        plt.xlabel('Time [s]',);
        plt.ylabel(r'$\lambda_{\theta}$ $[m^{2}/(s^{2}\cdot rad)]$',);
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.subplot(223)
        plt.plot(sol.x*sol.p,sol.y[6])
        plt.xlabel('Time [s]',);
        plt.ylabel(r'$\lambda_{v}$ $[m/s]$',);
        
        
        plt.subplot(224)
        plt.plot(sol.x*sol.p,sol.y[7])
        plt.xlabel('Time [s]',);
        plt.ylabel(r'$\lambda_{\gamma}$ $[m^{2}/(s^{2}\cdot rad)$');
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    plt.show()
    

"""
plt.figure(figsize=(10,10))

for sol in a:
    
    x = sol[0]
    tf = x[-1]
    x[-1] = 1
    x = x*tf
    
    dr = sol[2]*model.constant['R0']/1000
    
    dr = dr - dr[-1] + 100
    
    plt.subplot(221)
    plt.plot(dr,(sol[1]-model.constant['R0'])/1000)
    plt.xlabel('Downrange [km]',)
    plt.ylabel('Altitude [km]',)
    
    plt.subplot(222)
    plt.plot(x,sol[3])
    plt.xlabel('Time [s]',);
    plt.ylabel('Velocity [m/s]',);
    
    plt.subplot(223)
    plt.plot(x,sol[4])
    plt.xlabel('Time [s]',);
    plt.ylabel('Flight Path Angle [rad]',);
    
    plt.subplot(224)
    plt.plot(x,sol[-1])
    plt.xlabel('Time [s]',);
    plt.ylabel('Optimal Control alpha [deg]',);
plt.show()

plt.figure()

for sol in a:
    
    x = sol[0]
    tf = x[-1]
    x[-1] = 1
    x = x*tf
    
    dr = sol[2]*model.constant['R0']/1000
    
    dr = dr - dr[-1] + 100
    
    plt.subplot(221)
    plt.plot(dr,(sol[1]-model.constant['R0'])/1000)
    plt.xlabel('Downrange [km]',)
    plt.ylabel('Altitude [km]',)
    
    plt.subplot(222)
    plt.plot(dr,sol[3])
    plt.xlabel('Downrange [km]',);
    plt.ylabel('Velocity [m/s]',);
    
    plt.subplot(223)
    plt.plot(dr,sol[4])
    plt.xlabel('Downrange [km]',)
    plt.ylabel('Flight Path Angle [rad]',);
    
    plt.subplot(224)
    plt.plot(dr,sol[-1])
    plt.xlabel('Downrange [km]',)
    plt.ylabel('Optimal Control alpha [deg]',);
plt.show()



"""     
     
    