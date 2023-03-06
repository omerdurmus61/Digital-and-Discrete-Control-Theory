import numpy as np
import matplotlib.pyplot as plt
import scipy

class Solver(object):


    # Capability of this solver
    #
    # x_dot = a*x + b*u
    #
    # example for forward difference
    #
    #   x[k+1] - x[k]
    # ---------------- = a*x[k] + b*u[k]
    #        dt
    #

    def __init__(self,dt,initial_condition,initial_condition_2,t_initial,t_final,system_coef_A,system_coef_B,u):
        self.dt = dt
        self.initial_condition = initial_condition
        self.initial_condition_2 = initial_condition_2
        self.t_initial = t_initial
        self.t_final = t_final
        self.system_coef_A = system_coef_A
        self.system_coef_B = system_coef_B
        self.u = u
    
    def forward_difference(self):
        t = np.arange(self.t_initial,self.t_final,self.dt)
        x_t_f = np.arange(self.t_initial,self.t_final,self.dt)
        x_t_f[0] = self.initial_condition
        for k in range(0,int((t_final-t_initial)/dt) -1 ):
            x_t_f[k+1] = x_t_f[k] + self.dt*(self.system_coef_A*x_t_f[k]+self.system_coef_B*u[k])
        
        return x_t_f,t

    def backward_difference(self):
        t = np.arange(self.t_initial,self.t_final,self.dt)
        x_t_b = np.arange(self.t_initial,self.t_final,self.dt)
        x_t_b[0] = self.initial_condition
        for k in range(1,int((t_final-t_initial)/dt)):
            x_t_b[k] = x_t_b[k-1] + (self.dt*(self.system_coef_A*x_t_b[k-1]+self.system_coef_B*u[k]))

        
        return x_t_b,t

    def centered_difference(self):
        t = np.arange(self.t_initial,self.t_final,self.dt)
        x_t_c = np.arange(self.t_initial,self.t_final,self.dt)
        x_t_c[0] = self.initial_condition
        x_t_c[1] = self.initial_condition_2
        for k in range(1,int((t_final-t_initial)/dt -1)):
            x_t_c[k+1] = x_t_c[k-1] + 2*self.dt*(self.system_coef_A*x_t_c[k-1]+self.system_coef_B*u[k])

        return x_t_c,t




dt = 0.01

t_initial = 0.0
t_final = 1.0

system_coef_A = -5
system_coef_B = 0

x_0 = 10

u = np.ones(int((t_final-t_initial)/dt))*10

# Data for plotting
t = np.arange(t_initial,t_final,dt)
x_t_f = np.arange(t_initial,t_final,dt)
x_t_b = np.arange(t_initial,t_final,dt)
x_t_c = np.arange(t_initial,t_final,dt)


x_t_f[0] = x_0
x_t_b[0] = x_0
x_t_c[0] = x_0
x_t_c[1] = x_0 + system_coef_A*dt*x_0/5

# forward difference
for k in range(0,int((t_final-t_initial)/dt) -1 ):
    x_t_f[k+1] = x_t_f[k] + dt*system_coef_A*x_t_f[k]
  


# backward difference
for k in range(1,int((t_final-t_initial)/dt)):
    x_t_b[k] = x_t_b[k-1] +  dt*system_coef_A*x_t_b[k-1]



# centered difference
for k in range(1,int((t_final-t_initial)/dt -1)):
    x_t_c[k+1] = x_t_c[k-1] + 2*dt*system_coef_A*x_t_c[k-1]

plt.figure()
plt.subplot(311)
plt.plot(t,x_t_f,"r")
plt.grid(True)
plt.ylabel("Meter (m)")
plt.title("Forward solution")

plt.subplot(312)
plt.plot(t,x_t_b,"r")
plt.grid(True)
plt.ylabel("Meter (m)")
plt.title("Backward solution")

plt.subplot(313)
plt.plot(t,x_t_c,"r")
plt.grid(True)
plt.ylabel("Meter (m)")
plt.title("Centered solution")
plt.xlabel("Time (sec) ")
plt.show()