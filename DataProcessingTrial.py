import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

cell_concentration = pd.read_csv('./data/raman_data.csv')

labels = np.array(cell_concentration['condition'])
frequency_list = np.array(cell_concentration.columns)[1:]
features = np.array(cell_concentration.iloc[:,1:])

np.shape(frequency_list)

import numpy as np
import scipy.signal
from sklearn.decomposition import PCA

def pre_processing(x, smoothening, normalization, feature_red):
    '''
    Function to implement pre-processing of the data
    
    Input:
        x            : Data to be pre-processed
        smoothening  : {0:'No smoothening', 1: 'Moving Average Filter', 2: 'Savitzky Golay Filter'}
        normalization: {0: 'No normalization', 1: 'Normalize'}
        feature_red  : {0:'No reduction', 1:'PCA', 2:'Manual Reduction'}
    '''
    if smoothening:
        if smoothening == 1:
            N = 50
            x = np.convolve(x, np.ones(N)/N, mode='valid')
        elif smoothening == 2:
            x = scipy.signal.savgol_filter(x, 51, 2)
            
    if normalization:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        
    if feature_red:
        if feature_red == 1:
            # Assuming x is a 2D array where each row is a feature vector
            pca = PCA(n_components=0.95)  # Retain 95% of variance
            x = pca.fit_transform(x)
        elif feature_red == 2:
            # Implement manual feature reduction here
            # For example, selecting the first N features
            N = 10
            x = x[:, :N]
    
    return x

np.min((features-np.min(features,0))/(np.max(features,0)-np.min(features,0)))
cell_concentration.sample(5)
plt.hist(cell_concentration['condition'])
plt.title('Distribution of classes')
plt.xlabel('Cell Concentration')
plt.ylabel('Frequency')
plt.show()

fig, axs = plt.subplots(2,2)

x = np.array(cell_concentration.columns)[1:].astype(float)
a = 0
b = 0

temp_ind = np.random.choice(np.where(cell_concentration['condition']=='0mM')[0])
axs[0, 0].plot(x, np.array(cell_concentration.iloc[temp_ind,:])[1:])
axs[0, 0].set_title('0mM')

temp_ind = np.random.choice(np.where(cell_concentration['condition']=='0.1mM')[0])
axs[0, 1].plot(x, np.array(cell_concentration.iloc[temp_ind,:])[1:])
axs[0, 1].set_title('0.1mM')


temp_ind = np.random.choice(np.where(cell_concentration['condition']=='0.5mM')[0])
axs[1, 0].plot(x, np.array(cell_concentration.iloc[temp_ind,:])[1:])
axs[1, 0].set_title('0.5mM')


temp_ind = np.random.choice(np.where(cell_concentration['condition']=='1mM')[0])
axs[1, 1].plot(x, np.array(cell_concentration.iloc[temp_ind,:])[1:])
axs[1, 1].set_title('1mM')

for ax in axs.flat:
    ax.label_outer()
    
fig.set_figwidth(20)
fig.set_figheight(20)

fig.suptitle('Data Visualization',fontsize=26)

plt.show()

temp_x = np.array(cell_concentration.columns)[1:].astype(float)
x = temp_x[np.where(np.logical_and(temp_x>=800, temp_x<=1800))]

np.logical_or(temp_x>=800, temp_x<=1800)

fig, axs = plt.subplots(2,2)

temp_x = np.array(cell_concentration.columns)[1:].astype(float)
x = temp_x[np.where(np.logical_and(temp_x>=800, temp_x<=1800))]

a = 0
b = 0

temp_ind = np.random.choice(np.where(cell_concentration['condition']=='0mM')[0])
axs[0, 0].plot(x, np.array(cell_concentration.loc[temp_ind,x.astype(str)]))
axs[0, 0].set_title('0mM')

temp_ind = np.random.choice(np.where(cell_concentration['condition']=='0.1mM')[0])
axs[0, 1].plot(x, np.array(cell_concentration.loc[temp_ind,x.astype(str)]))
axs[0, 1].set_title('0.1mM')


temp_ind = np.random.choice(np.where(cell_concentration['condition']=='0.5mM')[0])
axs[1, 0].plot(x, np.array(cell_concentration.loc[temp_ind,x.astype(str)]))
axs[1, 0].set_title('0.5mM')


temp_ind = np.random.choice(np.where(cell_concentration['condition']=='1mM')[0])
axs[1, 1].plot(x, np.array(cell_concentration.loc[temp_ind,x.astype(str)]))
axs[1, 1].set_title('1mM')

for ax in axs.flat:
    ax.label_outer()
    
fig.set_figwidth(20)
fig.set_figheight(20)

fig.suptitle('Manually Filtered Data Visualization',fontsize=26)

plt.show()

