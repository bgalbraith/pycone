from __future__ import division
from brian import *
import numpy as np
import scipy as sp

# Input
dt = 1
T  = 1000
N  = 100

mu = np.linspace(0,np.pi,N)
#mu = np.random.rand(N)*np.pi

# Parameters
Cm  = 1*ufarad
EL  = -60*mV
EK  = -90*mV
ENa = 50*mV
gL  = 5e-5*siemens
gNa = 10*msiemens
gK  = 30*msiemens
VT  = -63*mV

# The HH model
eqs=Equations('''
dv/dt  = (gL*(EL - v) + Iext - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK))/Cm : volt
dm/dt  = alpham*(1 - m) - betam*m : 1
dn/dt  = alphan*(1 - n) - betan*n : 1
dh/dt  = alphah*(1 - h) - betah*h : 1
alpham = 0.32*(mV**-1)*(13*mV - v + VT)/(exp((13*mV - v + VT)/(4*mV)) - 1)/ms : Hz
betam  = 0.28*(mV**-1)*(v - VT - 40*mV)/(exp((v - VT - 40*mV)/(5*mV)) - 1)/ms : Hz
alphah = 0.128*exp((17*mV - v + VT)/(18*mV))/ms : Hz
betah  = 4 / (1 + exp((40*mV - v + VT)/(5*mV)))/ms : Hz
alphan = 0.032*(mV**-1)*(15*mV - v + VT)/(exp((15*mV - v + VT)/(5*mV)) - 1)/ms : Hz
betan  = .5*exp((10*mV - v + VT)/(40*mV))/ms : Hz
Iext : uA
''')

# The network
P=NeuronGroup(N,
              model     = eqs,
              threshold = EmpiricalThreshold(threshold=-20*mV, refractory=2*ms),
              implicit  = True,
              freeze    = True)

# Initialization
P.v    = EL*mV
P.Iext = 0*uA

#tau_ref  = 2*ms
#tau_rc   = 20*ms
#min_resp = 10*Hz
#max_resp = 100*Hz
#J_bias   = 1/(1 - exp((tau_ref*min_resp - 1)/(tau_rc*min_resp)))
#alpha    = 1/(1 - exp((tau_ref*max_resp - 1)/(tau_rc*max_resp))) - J_bias

myclock = Clock(dt=1*ms)
@network_operation(myclock, when='start')
def updateInput(clock):
  index  = int(clock.t/clock.dt)
  #P.Iext = alpha*x[index]*uA + J_bias*uA
  P.Iext = sp.stats.vonmises.pdf(np.pi/2-mu,2)*uA
  
# Record a few trace
trace  = StateMonitor(P, 'v', record=[49])
spikes = SpikeMonitor(P)
run(1000*msecond)
raster_plot(spikes)
plot(trace.times/ms,trace[49]/mV)
#plot(-95+5*x)
ylabel('Membrane Potential (mV)')
show()
