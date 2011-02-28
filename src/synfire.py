from __future__ import division
import numpy as np
import pylab as py

## Neuron model parameters
# time constants (ms)
ex_rise  = 1
ex_decay = 5
in_rise  = 1
#in_Cl    = 5
#in_K     = 20
#ns_decay = 5
tau_m    = 5
#tau_K    = 20
tau_ref  = 3

# potentials (mV)
Vrest = 0
Vth   = 14


## Network parameters
pools     = 1
pool_size = 1
con_intra = 15
con_inter = 2

## Sim parameters
T = 20 #ms
dt = 0.01 # ms
time = np.arange(0,T+dt,dt)

N  = pools * pool_size
Nt = len(time)

Vm = np.zeros([N,Nt])
I  = np.zeros([N,Nt])
syn = np.zeros([N,Nt])
inh = np.zeros(Nt)
raster = np.zeros([N,Nt])
raster[:] = np.nan
last_spike = np.ones(N)*-tau_ref

Iext = np.zeros([N,Nt])
Iext[0:30,:] = 20

A = np.zeros([N,N])
for p in range(pools):
    A[p*pool_size:(p+1)*pool_size,p*pool_size:(p+1*pool_size)] = 0.8

#soma = lambda V: gK*(V - Ek) + gNa*(V - ENa) + gCl*(V - ECl)

# this can probably be precomputed
def syncurrent(t):
    if t < 0:
        t = 0
    if t < ex_rise:
        return t*np.exp(-t/ex_rise)
    else:
        return ex_rise/np.exp(1)*np.exp(-((t - ex_rise)/ex_decay))

Iden = np.vectorize(syncurrent)

for i, t in enumerate(time):
    syn[:,i] = Iden(t - last_spike)
    I[:,i]   = syn[:,i] + Iext[:,i]

    if i == 0:
        continue
    active = np.nonzero(t > last_spike + tau_ref)
    Vm[active,i] = Vm[active,i-1] + dt * (-Vm[active,i-1] + I[active,i-1]) / tau_m
    spiked = np.nonzero(Vm[:,i] >= Vth)
    raster[spiked,i] = spiked[0]
    last_spike[spiked] = t
    Vm[spiked,i] = 0

    #inh[i] = inh[i-1] + dt * (-inh[i-1] + sum(syn[i-1],0)) / in_rise



py.figure()
py.plot(time,raster[0,:],'.')
py.figure()
py.plot(time,Vm[0,:])
py.figure()
py.plot(time,I[0,:])
py.figure()
py.plot(time,syn[0,:])
py.show()