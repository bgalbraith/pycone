from numpy import *
from pylab import *

################################################################################
# Classes
################################################################################
class IzhNeuron:
  def __init__(self, label, a, b, c, d, v0, u0=None):
    self.label = label

    self.a = a
    self.b = b
    self.c = c
    self.d = d
	
    self.v = v0
    self.u = u0 if u0 is not None else b*v0

	
class IzhSim:
  def __init__(self, n, T, dt=0.25):
    self.neuron = n
    self.dt     = dt
    self.t      = t = arange(0, T+dt, dt)
    self.stim   = zeros(len(t))
    self.x      = 5
    self.y      = 140
    self.du     = lambda a, b, v, u: a*(b*v - u)
	
  def integrate(self, n=None):
    if n is None: n = self.neuron
    trace = zeros((2,len(self.t)))
    for i, j in enumerate(self.stim):
      n.v += self.dt * (0.04*n.v**2 + self.x*n.v + self.y - n.u + self.stim[i])
      n.u += self.dt * self.du(n.a,n.b,n.v,n.u)
      if n.v > 30:
        trace[0,i] = 30
        n.v        = n.c
        n.u       += n.d
      else:
        trace[0,i] = n.v
        trace[1,i] = n.u
    return trace

################################################################################
# Models
################################################################################
sims = []

## (A) tonic spiking
n = IzhNeuron("(A) tonic spiking", a=0.02, b=0.2, c=-65, d=6, v0=-70)
s = IzhSim(n, T=100)
for i, t in enumerate(s.t):
  s.stim[i] = 14 if t > 10 else 0
sims.append(s)

## (B) phasic spiking
n = IzhNeuron("(B) phasic spiking", a=0.02, b=0.25, c=-65, d=6, v0=-64)
s = IzhSim(n, T=200)
for i, t in enumerate(s.t):
  s.stim[i] = 0.5 if t > 20 else 0
sims.append(s)

## (C) tonic bursting
n = IzhNeuron("(C) tonic bursting", a=0.02, b=0.25, c=-50, d=2, v0=-70)
s = IzhSim(n, T=220)
for i, t in enumerate(s.t):
  s.stim[i] = 15 if t > 22 else 0
sims.append(s)

## (D) phasic bursting
n = IzhNeuron("(D) phasic bursting", a=0.02, b=0.25, c=-55, d=0.05, v0=-70)
s = IzhSim(n, T=200)
for i, t in enumerate(s.t):
  s.stim[i] = 0.6 if t > 20 else 0
sims.append(s)

## (E) mixed mode
n = IzhNeuron("(E) mixed mode", a=0.02, b=0.2, c=-55, d=4, v0=-70)
s = IzhSim(n, T=160)
for i, t in enumerate(s.t):
  s.stim[i] = 10 if t > 16 else 0
sims.append(s)

## (F) spike freq. adapt
n = IzhNeuron("(F) spike freq. adapt", a=0.01, b=0.2, c=-65, d=8, v0=-70)
s = IzhSim(n, T=85)
for i, t in enumerate(s.t):
  s.stim[i] = 30 if t > 8.5 else 0
sims.append(s)

## (G) Class 1 exc.
n = IzhNeuron("(G) Class 1 exc.", a=0.02, b=-0.1, c=-55, d=6, v0=-60)
s = IzhSim(n, T=300)
s.x = 4.1
s.y = 108
for i, t in enumerate(s.t):
  s.stim[i] = 0.075*(t-30) if t > 30 else 0
sims.append(s)

## (H) Class 2 exc.
n = IzhNeuron("(H) Class 2 exc.", a=0.2, b=0.26, c=-65, d=0, v0=-64)
s = IzhSim(n, T=300)
for i, t in enumerate(s.t):
  s.stim[i] = -0.5+(0.015*(t-30)) if t > 30 else -0.5
sims.append(s)

## (I) spike latency
n = IzhNeuron("(I) spike latency", a=0.02, b=0.2, c=-65, d=6, v0=-70)
s = IzhSim(n, T=100, dt=0.2)
for i, t in enumerate(s.t):
  s.stim[i] = 7.04 if 10 < t < 13 else 0
sims.append(s)

## (J) subthresh. osc.
n = IzhNeuron("(J) subthresh. osc.", a=0.05, b=0.26, c=-60, d=0, v0=-62)
s = IzhSim(n, T=200)
for i, t in enumerate(s.t):
  s.stim[i] = 2 if 20 < t < 25 else 0
sims.append(s)

## (K) resonator
n = IzhNeuron("(K) resonator", a=0.1, b=0.26, c=-60, d=-1, v0=-62)
s = IzhSim(n, T=400)
Ts  = array([40,60,280,320])
for i, t in enumerate(s.t):
  s.stim[i] = 0.65 if ((Ts < t) & (t < Ts+4)).any() else 0
sims.append(s)

## (L) integrator
n = IzhNeuron("(L) integrator", a=0.02, b=-0.1, c=-55, d=6, v0=-60)
s = IzhSim(n, T=100)
s.x = 4.1
s.y = 108
Ts  = array([9.09,14.09,70,80])
for i, t in enumerate(s.t):
  s.stim[i] = 9 if ((Ts < t) & (t < Ts+2)).any() else 0
sims.append(s)

## (M) rebound spike
n = IzhNeuron("(M) rebound spike", a=0.03, b=0.25, c=-60, d=4, v0=-64)
s = IzhSim(n, T=200, dt=0.2)
for i, t in enumerate(s.t):
  s.stim[i] = -15 if 20 < t < 25 else 0
sims.append(s)

## (N) rebound burst
n = IzhNeuron("(N) rebound burst", a=0.03, b=0.25, c=-52, d=0, v0=-64)
s = IzhSim(n, T=200, dt=0.2)
for i, t in enumerate(s.t):
  s.stim[i] = -15 if 20 < t < 25 else 0
sims.append(s)

## (O) thresh. variability
n = IzhNeuron("(O) thresh. variability", a=0.03, b=0.25, c=-60, d=4, v0=-64)
s = IzhSim(n, T=100)
for i, t in enumerate(s.t):
  if (10 < t < 15) or (80 < t < 85): s.stim[i] = 1
  elif 70 < t < 75:                  s.stim[i] = -6
  else:                              s.stim[i] = 0
sims.append(s)

## (P) bistability
n = IzhNeuron("(P) bistability", a=0.1, b=0.26, c=-60, d=0, v0=-61)
s = IzhSim(n, T=300)
for i, t in enumerate(s.t):
  s.stim[i] = 1.24 if (37.5 < t < 42.5) or (216 < t < 221) else 0.24
sims.append(s)

## (Q) DAP
n = IzhNeuron("(Q) DAP", a=1, b=0.2, c=-60, d=-21, v0=-70)
s = IzhSim(n, T=50, dt=0.1)
for i, t in enumerate(s.t):
  s.stim[i] = 20 if -1 < (t - 10) < 1 else 0
sims.append(s)

## (R) accomodation
n = IzhNeuron("(R) accomodation", a=0.02, b=1, c=-55, d=4, v0=-65, u0=-16)
s = IzhSim(n, T=400, dt=0.5)
s.du = lambda a, b, v, u: a*(b*(v + 65))
for i, t in enumerate(s.t):
  if t < 200:     s.stim[i] = t/25
  elif t < 300:   s.stim[i] = 0
  elif t < 312.5: s.stim[i] = (t-300)/12.5*4
  else:           s.stim[i] = 0
sims.append(s)

## (S) inhibition induced spiking
n = IzhNeuron("(S) inhibition induced spiking", a=-0.02, b=-1, c=-60, d=8, v0=-63.8)
s = IzhSim(n, T=350, dt=0.5)
for i, t in enumerate(s.t):
  s.stim[i] = 80 if t < 50 or t > 250 else 75
sims.append(s)

## (T) inhibition induced bursting
n = IzhNeuron("(T) inhibition induced bursting", a=-0.026, b=-1, c=-45, d=-2, v0=-63.8)
s = IzhSim(n, T=350, dt=0.5)
for i, t in enumerate(s.t):
  s.stim[i] = 80 if t < 50 or t > 250 else 75
sims.append(s)

################################################################################
# Simulate
################################################################################
fig = figure()
fig.set_title('Izhikevich Examples')
for i,s in enumerate(sims):
  res = s.integrate()
  ax  = subplot(5,4,i+1)

  ax.plot(s.t, res[0], s.t, -95 + ((s.stim - min(s.stim))/(max(s.stim) - min(s.stim)))*10)

  ax.set_xlim([0,s.t[-1]])
  ax.set_ylim([-100, 35])
  ax.set_title(s.neuron.label, size="small")
  ax.set_xticklabels([])
  ax.set_yticklabels([])
show()
