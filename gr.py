# -*- coding: utf-8 -*-
"""
Created on Sun Aug 09 13:48:40 2015

@author: Walter
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd

import pymc3 as mc
import theano.tensor as t

import matplotlib.pyplot as plt
from scipy import optimize



#import powerlaw


gr1 = pd.read_csv('gr_eta.05000', sep='\t', header=None)
gr2 = pd.read_csv('gr_eta.05500', sep='\t', header=None)
gr3 = pd.read_csv('gr_eta.06000', sep='\t', header=None)
gr4 = pd.read_csv('gr_eta.06500', sep='\t', header=None)
gr5 = pd.read_csv('gr_eta.07000', sep='\t', header=None)
gr6 = pd.read_csv('gr_eta.07500', sep='\t', header=None)

plt.scatter(np.log(gr1[0]), np.log(gr1[1]))
plt.scatter(np.log(gr2[0]), np.log(gr2[1]))
plt.scatter(np.log(gr3[0]), np.log(gr3[1]))
plt.scatter(np.log(gr4[0]), np.log(gr4[1]))
plt.scatter(np.log(gr5[0]), np.log(gr5[1]))
plt.scatter(np.log(gr6[0]), np.log(gr6[1]))
plt.show()

cutoff = 0.5

gr1 = gr1[gr1[0]<cutoff]
gr2 = gr2[gr2[0]<cutoff]
gr3 = gr3[gr3[0]<cutoff]
gr4 = gr4[gr4[0]<cutoff]
gr5 = gr5[gr5[0]<cutoff]
gr6 = gr6[gr6[0]<cutoff]

plt.scatter(np.log(gr1[0]), np.log(gr1[1]))
plt.scatter(np.log(gr2[0]), np.log(gr2[1]))
plt.scatter(np.log(gr3[0]), np.log(gr3[1]))
plt.scatter(np.log(gr4[0]), np.log(gr4[1]))
plt.scatter(np.log(gr5[0]), np.log(gr5[1]))
plt.scatter(np.log(gr6[0]), np.log(gr6[1]))

gr = gr1
gr = np.concatenate((gr, gr2))
gr = np.concatenate((gr, gr3))
gr = np.concatenate((gr, gr4))
gr = np.concatenate((gr, gr5))
gr = np.concatenate((gr, gr6))

plt.scatter(np.log(gr[:,0]), np.log(gr[:,1]))
plt.show()

with mc.Model() as model:
  
  r = gr[:,0]
  g = gr[:,1]
  
  c0 = mc.Exponential('c0', lam=1)
  c1 = mc.Exponential('c1', lam=1)
  s = mc.HalfNormal('s', sd=10)
  
  mu = mc.Deterministic('mu', c0 * t.pow((1./r), c1))
 
#  gr_sim = mc.Normal('gr_sim', mu=mu, sd=s, observed=g) 
  gr_sim = mc.T('gr_sim', nu=1, mu=mu, lam=s, observed=g)
  
  start = mc.find_MAP(model=model, fmin=optimize.fmin_powell)
  step = mc.NUTS(scaling=start)
  trace = mc.sample(5000, step, start=start)
  
mc.traceplot(trace, vars=['c0', 'c1', 's', 'mu']);

#
# Plot individual realizations against the raw data
#
a = trace['c0']
b = trace['c1']
x = np.linspace(0.025,0.975,100)

fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(1,2,1)

plt.scatter(r, g, color='g', alpha=0.5, s=50)
for i in xrange(500):
  y = a[i] * (1./x) ** b[i]
  plt.plot(x, y, 'b', alpha=0.01)

plt.xlim((0,cutoff))
plt.ylim((0,30))

ax = fig.add_subplot(1,2,2)

plt.scatter(r, g, color='g', alpha=0.5, s=50)
for i in xrange(500):
  y = a[i] * (1./x) ** b[i]
  plt.plot(x, y, 'b', alpha=0.01)

ax.set_yscale('log')
ax.set_xscale('log')

plt.xlim((0,cutoff))
plt.ylim((0,30))
