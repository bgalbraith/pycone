import numpy as np
import pylab as py

## setup parameters and state variables
T      = 100                      # total time to simulate (msec)
dt     = 0.125                   # simulation time step (msec)
time   = np.arange(0, T+dt, dt)  # time array
N      = 20                      # number of neurons


## LIF properties
Vm      = np.zeros([N,len(time)])  # potential (V) trace over time
tau_m   = 10                    # time constant (msec)
tau_ref = 4                        # refractory period (msec)
tau_psc = 5                        # post synaptic current filter time constant
Vth     = 1                        # spike threshold (V)

## Input stimulus
Iext = np.zeros(N)
Iext[0] = 1.5

synapses = np.eye(N)
synapses = np.roll(synapses, 1, 0)

#synapses = np.random.rand(N,N)

last_spike = np.zeros(N)             # initial refractory time
raster = np.zeros([N,len(time)])*np.nan

def syncurrent(t):
    if t < 0:
        t = 0
    return t*np.exp(-t/tau_psc)
Isyn = np.vectorize(syncurrent)

I = np.zeros((N,len(time)))
## iterate over each time step
for i, t in enumerate(time):
    if i == 0:
        continue

    active = np.nonzero(t > last_spike + tau_ref)
    Vm[active,i] = Vm[active,i-1] + (-Vm[active,i-1] + I[active,i-1]) / tau_m * dt

    spiked = np.nonzero(Vm[:,i] > Vth)
    last_spike[spiked] = t
    raster[spiked,i] = spiked[0]+1
    I[:,i] = Iext + synapses.dot(Isyn(t - last_spike))

## plot membrane potential trace
#py.plot(time, np.transpose(Vm))
py.plot(time, np.transpose(raster), 'b.')
#py.plot(time,np.transpose(I))
py.title('Leaky Integrate-and-Fire Example')
py.ylabel('Membrane Potential (V)')
py.xlabel('Time (msec)')
py.show()
