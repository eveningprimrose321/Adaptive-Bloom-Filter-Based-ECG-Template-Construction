# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:19:40 2023

@author: ess601
"""

import numpy as np
from scipy.io import loadmat
from itertools import combinations
file = loadmat('./ptb_data/beatbundle_PTB_p_285.mat')
beatbundle = file['beatbundle']

data = np.zeros((285,50,1000))
for person in range (285):
    for beats in range (50):
        data[person,beats,:] = beatbundle[person,beats][:]
data += 5000
      
#%%

def bloom_filter(data):
    to_binary = np.vectorize(lambda x: bin(x)[2:])
    template = np.zeros((10,256))
    for part in range(10):
        datax = data[100*part:100*(part+1)].astype(int)
        datax_bin = to_binary(datax)
        feature = np.zeros((100,16))
        for bit in range (100):
            string = datax_bin[bit]
            char_list = [char for char in string]
            a = len(char_list)
            b = 16-a
            feature[bit,b:] = char_list
        feature = np.reshape(feature,(200,8))
        for row in range(200):
            d_0 = 0
            for bit in range(8):
                temp = (2**(7-bit))*feature[row,bit]
                d_0 += temp
            template[part,int(d_0)] += 1
    return np.reshape(template,(-1))

data_EN = data[:235,:,:]
data_OUT = data[235:,:,:]

#%%
#template construction

avg_idx = list(combinations([1, 2, 3, 4, 0], 3))
template = []
label = []
for p in range(235):
    template_per_person = []
    for idx in avg_idx:
        temp_ecg = np.mean(data_EN[p,idx,:],axis=0)
        template_per_person.append(bloom_filter(temp_ecg))
    template.append(np.mean(np.array(template_per_person),axis = 0))
    label.append(p)
                    
template = np.array(template)
label = np.array(label)
    
from sklearn.neighbors import KNeighborsClassifier 

Classifier = KNeighborsClassifier(n_neighbors=1)
Classifier.fit(template, label)

#%%
#iqr measurement
avg_idx = np.random.randint(5, 30, (20,3))
template_iqr = np.zeros((235,20,2560))
mse_all = np.zeros((235,20))
for p in range(235):
    for i,idx in enumerate(avg_idx):
        temp_ecg = np.mean(data_EN[p,idx,:],axis=0)
        template_iqr[p,i,:] = bloom_filter(temp_ecg)
        mse_all[p,i] = np.mean((template_iqr[p,i,:]-template[p,:])**2)
        
iqr = np.zeros((235,3)) #q1,q3,iqr
for p in range(235):
    iqr[p,0], iqr[p,1] = np.percentile(mse_all[p,:],[25,75])
    iqr[p,2] = iqr[p,1] - iqr[p,0]
    
#%%
#enrollee idenfication
avg_idx = np.random.randint(30, 50, (20,3))
template_test = []
label_test = []
for p in range(235):
    for idx in avg_idx:
        temp_ecg = np.mean(data_EN[p,idx,:],axis=0)
        template_test.append(bloom_filter(temp_ecg))
        label_test.append(p)
template_test = np.array(template_test)
predict_test = Classifier.predict(template_test)
IR = sum(predict_test==np.array(label_test))/len(predict_test)

#%%
#outlier idenfication
avg_idx = np.random.randint(0, 50, (20,3))
template_out = []
for p in range(50):
    for idx in avg_idx:
        temp_ecg = np.mean(data_OUT[p,idx,:],axis=0)
        template_out.append(bloom_filter(temp_ecg))
template_out = np.array(template_out)
predict_out = Classifier.predict(template_out)

#%%
#verification
fpir_all = []
fnir_all = []

for k in np.arange(0,10,0.5):
    threshold = iqr[:,1] + k*iqr[:,2]
    fn, fp = 0, 0
    for i,pred in enumerate(predict_test):
        if pred != label_test[i]:
            fn += 1
        elif np.mean((template_test[i,:]-template[pred,:])**2) >= threshold[pred]:
            fn += 1
    for i, pred in enumerate(predict_out):
        if np.mean((template_out[i,:]-template[pred,:])**2) < threshold[pred]:
            fp += 1
    fpir_all.append(fp/len(predict_out))
    fnir_all.append(fn/len(predict_test))


#%% plot
import matplotlib.pyplot as plt

#plt FNIR,FPIR
x =np.arange(0,10,0.5)

plt.plot(x,fnir_all,color='#800080',label= 'FNIR')
plt.plot(x,fpir_all,color='#6A5ACD',label= 'FPIR')

#plt EER
idx = 0
for i in range(len(x)):
    if fnir_all[i] < fpir_all[i]:
        idx = i
        break;
eer_x = round(idx*0.5,4)
eer_y = round(fnir_all[idx],4)
plt.plot(eer_x, eer_y, marker='o', markersize=8, color='#6495ed', label='EER')
plt.text(eer_x, eer_y+0.0125, f'({eer_x}, {eer_y})', fontsize=14, color='black', ha='center')


plt.xlabel('K',fontsize=12)
plt.ylabel('Percentage',fontsize=12)
plt.legend(fontsize=12)
plt.show()