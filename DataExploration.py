import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cell_concentration = pd.read_csv('./data/raman_data.csv')

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

