#!/usr/bin/python
#****************************************************************#
# ScriptName: helloworld.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-04-09 15:46
# Modify Author: fancangning.fcn@alibaba-inc.com
# Modify Date: 2021-04-09 15:46
# Function: draw picture of normal distribtion and sigmoid
#***************************************************************#

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams.update({'font.size':30})

def sigmoid(x):
    return 1. / ( 1. + np.exp(-x))

a = np.random.normal(0, 1, size=10000)
b = np.random.normal(4, 1, size=10000)
c = np.random.normal(8, 1, size=10000)

df = pd.DataFrame(
                {
                    'source_normal': a,
                    'target_normal': b,
                    'target_abnormal': c
                }
            )

fig, ax = plt.subplots()
sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
plt.xlabel('Reconstruction Loss')
plt.ylabel('Density')
plt.xlim(-5, 20)
plt.xticks([])
plt.yticks([])
fig.savefig('normal_distribution.png')

x = np.arange(-5, 13, 0.01)
y = sigmoid(-1*x+6)

fig, ax = plt.subplots()
plt.plot(x, y)
plt.xlabel('Reconstruction Loss')
plt.ylabel('Weight')
plt.xlim(-5, 20)
plt.xticks([])
plt.yticks([])
fig.savefig('sigmoid.png')