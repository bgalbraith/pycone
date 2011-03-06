from __future__ import division
import numpy as np
import pylab as py

## setup parameters and state variables
N    = 10                      # number of neurons
T    = 200                     # total time to simulate (msec)
dt   = 0.125                   # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array


## LIF properties
Vm      = np.zeros([N,len(time)])  # potential (V) trace over time
tau_m   = 10                       # time constant (msec)
tau_ref = 4                        # refractory period (msec)
tau_psc = 5                        # post synaptic current filter time constant
Vth     = 1                        # spike threshold (V)

## Currents
I    = np.zeros((N,len(time)))
Iext = np.zeros(N) # externally applied stimulus
Iext[0] = 1.5

## Synapse weight matrix
# equally weighted ring connectivity
synapses = np.eye(N)
synapses = np.roll(synapses, -1, 1)

# randomly weighted full connectivity
#synapses = np.random.rand(N,N)*0.3

## Synapse current model
def Isyn(t):
    '''t is an array of times since each neuron's last spike event'''
    t[np.nonzero(t < 0)] = 0
    return t*np.exp(-t/tau_psc)
last_spike = np.zeros(N) - tau_ref

## Simulate network
raster = np.zeros([N,len(time)])*np.nan
for i, t in enumerate(time[1:],1):
    active = np.nonzero(t > last_spike + tau_ref)
    Vm[active,i] = Vm[active,i-1] + (-Vm[active,i-1] + I[active,i-1]) / tau_m * dt

    spiked = np.nonzero(Vm[:,i] > Vth)
    last_spike[spiked] = t
    raster[spiked,i] = spiked[0]+1
    I[:,i] = Iext + synapses.dot(Isyn(t - last_spike))

## plot membrane potential trace
py.plot(time, np.transpose(raster), 'b.')
py.title('Recurrent Network Example')
py.ylabel('Neuron')
py.xlabel('Time (msec)')
py.ylim([0.75,N+0.25])
py.show()