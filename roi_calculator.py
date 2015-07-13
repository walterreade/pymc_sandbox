# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:07:19 2015

@author: Walter Reade
"""
from __future__ import division, print_function

import numpy as np
import pymc3 as mc

import matplotlib.pylab as plt
import seaborn as sns
sns.set()

pay = 2.85
cpv = 0.015
imp = 1000
clk = 100
con = 7

with mc.Model() as model:

  p_ctr = mc.Beta('p_ctr', clk+1, imp+1)
  p_cvr = mc.Beta('p_cvr', con+1, clk+1)

  m_clk = mc.Binomial('m_clk', imp, p_ctr, observed=clk)  
  m_con = mc.Binomial('m_con', m_clk, p_cvr, observed=con)
  
  mc.find_MAP()
  trace = mc.sample(10000, mc.NUTS())
  
  mc.traceplot(trace, vars=['p_ctr','p_cvr'])
  plt.show()



revenue = pay * trace['p_cvr'] * trace['p_ctr'] * imp
cost = cpv*imp
roi = (revenue - cost) / cost

y = int(sum(roi>0)/len(roi)*100)
title = 'Profitable? Yes: {}%, No: {}%'.format(y,100-y)

bins = 0.10 * np.arange(-10,50)
plt.hist(roi[np.where(roi>0)],bins=bins,color='g')
plt.hist(roi[np.where(roi<=0)],bins=bins,color='r')
plt.xlim(-1,np.ceil(roi.max()))
plt.title(title)
plt.show()
