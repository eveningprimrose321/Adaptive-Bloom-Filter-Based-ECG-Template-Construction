# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:19:40 2023

@author: ess601
"""

import numpy as np
from scipy.io import loadmat
from itertools import combinations
from itertools import combinations
file = loadmat('../../ptb_data/beatbundle_PTB_p_285.mat')
beatbundle = file['beatbundle']

data = np.zeros((285,50,1000))
for person in range (285):
    for beats in range (50):
        data[person,beats,:] = beatbundle[person,beats][:]
        
data_EN = data[:235,:,:] 
data_OUT = data[235:,:,:] 

max_value = np.max(data_EN) #5445.24
min_value = np.min(data_EN) #-1441.07
      
#%%
def bloom_filter(data):
    #Normalization
    NORMdata = 2048*((data - min_value)/(max_value-min_value))
    
    neg_count = 0
    for point in range(1000):
        if NORMdata[point] < 0:
            neg_count +=1
            NORMdata[point] = 0
    
    #Bloom Filter
    to_binary = np.vectorize(lambda x: bin(x)[2:])
    template = np.zeros((10,64))
    datax = np.zeros((100))
    for part in range(10):
        for lastdigit in range(100):
            datax[lastdigit] = NORMdata[lastdigit+(part*100)]
        datax = datax.astype(int)
        datax_bin = to_binary(datax)
        feature = np.zeros((100,12))
        for bit in range (100):
            string = datax_bin[bit]
            char_list = [char for char in string]
            a = len(char_list)
            b = 12-a
            feature[bit,b:] = char_list
        feature = np.reshape(feature,(200,6))
        for row in range(200):
            d_0 = 0
            for bit in range(6):
                temp = (2**(5-bit))*feature[row,bit]
                d_0 += temp
            template[part,int(d_0)] += 1 
    return np.reshape(template,(-1)), neg_count #-1:不確定有幾項(refer to 10)

#%%
#template construction

#neg_check=[]

avg_idx = np.random.randint(0, 10, (60,3))  #(60/120) 60可改

template = []
label = []
for p in range(235):
    template_per_person = []
    for idx in avg_idx:
        temp_ecg = np.mean(data_EN[p,idx,:],axis=0)
        temp = bloom_filter(temp_ecg)
        template_per_person.append(temp[0])
        #neg_check.append(temp[1])
    template.append(np.mean(np.array(template_per_person),axis = 0))
   
    label.append(p)
                    
template = np.array(template)
label = np.array(label)
    
from sklearn.neighbors import KNeighborsClassifier 

Classifier = KNeighborsClassifier(n_neighbors=1)
Classifier.fit(template, label)

#%%
#iqr measurement
from numpy.linalg import norm

avg_idx = np.random.randint(10, 30, (60,3))
template_iqr = np.zeros((235,60,640))
mse_all = np.zeros((235,60))
cosim_all = np.zeros((235,60))

for p in range(235):
    for i,idx in enumerate(avg_idx):
        temp_ecg = np.mean(data_EN[p,idx,:],axis=0)
        template_iqr[p,i,:] = bloom_filter(temp_ecg)[0]
        mse_all[p,i] = np.mean((template_iqr[p,i,:]-template[p,:])**2)
        cosim_all[p,i] = np.dot(template_iqr[p,i,:],template[p,:])/(norm(template_iqr[p,i,:])*norm(template[p,:]))
        
iqr_MSE = np.zeros((235,3)) #q1,q3,iqr
for p in range(235):
    iqr_MSE[p,0], iqr_MSE[p,1] = np.percentile(mse_all[p,:],[25,75])
    iqr_MSE[p,2] = iqr_MSE[p,1] - iqr_MSE[p,0]

iqr_CS = np.zeros((235,3)) #q1,q3,iqr
for p in range(235):
    iqr_CS[p,0], iqr_CS[p,1] = np.percentile(cosim_all[p,:],[25,75])
    iqr_CS[p,2] = iqr_CS[p,1] - iqr_CS[p,0]
    
#%%
#enrollee idenfication
avg_idx = np.random.randint(30, 50, (60,5))
template_test = []
label_test = []
for p in range(235):
    for idx in avg_idx:
        temp_ecg = np.mean(data_EN[p,idx,:],axis=0)
        template_test.append(bloom_filter(temp_ecg)[0])
        label_test.append(p)
template_test = np.array(template_test)
predict_test = Classifier.predict(template_test)
IR = sum(predict_test==np.array(label_test))/len(predict_test)
IR = round(IR,4)
#%%
#outlier idenfication
neg_check=[]
avg_idx = np.random.randint(0, 50, (60,5))
template_out = []
for p in range(50):
    for idx in avg_idx:
        temp_ecg = np.mean(data_OUT[p,idx,:],axis=0)
        temp = bloom_filter(temp_ecg)
        template_out.append(temp[0])
        neg_check.append(temp[1])
template_out = np.array(template_out)
predict_out = Classifier.predict(template_out)

#%%
#verification_mse
fpir_all_MSE = []
fnir_all_MSE = []

for k in np.arange(0,10,0.5):
    threshold = iqr_MSE[:,1] + k*iqr_MSE[:,2]
    fn, fp = 0, 0
    for i,pred in enumerate(predict_test):
        if pred != label_test[i]:
            fn += 1
        elif np.mean((template_test[i,:]-template[pred,:])**2) >= threshold[pred]:
            fn += 1
    for i, pred in enumerate(predict_out):
        if np.mean((template_out[i,:]-template[pred,:])**2) < threshold[pred]:
            fp += 1
    fpir_all_MSE.append(fp/len(predict_out))
    fnir_all_MSE.append(fn/len(predict_test))

#%%
#verification_Cosine Similarity
fpir_all_CS = []
fnir_all_CS = []
cosim_out = np.zeros((3000))

for k in np.arange(0,10,0.5):
    threshold = iqr_CS[:,0] - k*iqr_CS[:,2]
    fn, fp = 0, 0
    for i,pred in enumerate(predict_test):
        if pred != label_test[i]:
            fn += 1
        elif np.dot(template_test[i,:],template[pred,:])/(norm(template_test[i,:])*norm(template[pred,:])) <= threshold[pred]:
        #np.mean((template_test[i,:]-template[pred,:])**2) >= threshold[pred]:
            fn += 1
    for i, pred in enumerate(predict_out):
        cosim_out[i] = np.dot(template_out[i,:],template[pred,:])/(norm(template_out[i,:])*norm(template[pred,:]))
        if cosim_out[i] > threshold[pred]:
        #np.mean((template_out[i,:]-template[pred,:])**2) < threshold[pred]:
            fp += 1
    fpir_all_CS.append(fp/len(predict_out))
    fnir_all_CS.append(fn/len(predict_test))

#%% plot MSE
import matplotlib.pyplot as plt
plt.figure()
#plt FNIR,FPIR
x =np.arange(0,10,0.5)

plt.plot(x,fnir_all_MSE,color='#800080',label= 'FNIR')
plt.plot(x,fpir_all_MSE,color='#6A5ACD',label= 'FPIR')

#plt EER
idx = 0
for i in range(len(x)):
    if fnir_all_MSE[i] < fpir_all_MSE[i]:
        idx = i
        break;
x1 = idx*0.5
x2 = (idx-1)*0.5
y1 = fnir_all_MSE[idx]
y2 = fnir_all_MSE[idx-1]
y3 = fpir_all_MSE[idx]
y4 = fpir_all_MSE[idx-1]
eer_x = ((x1*y2-y1*x2)*(x1-x2)-(x1-x2)*(x1*y4-y3*x2))/((x1-x2)*(y3-y4)-(y1-y2)*(x1-x2))
eer_y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x1*y4-y3*x2))/((x1-x2)*(y3-y4)-(y1-y2)*(x1-x2))
eer_x = round(eer_x,4)
eer_y = round(eer_y,4)

#eer_x = round(idx*0.5,4)
#eer_y = round(fnir_all[idx],4)

plt.plot(eer_x, eer_y, marker='o', markersize=8, color='#6495ed', label='EER')
plt.text(eer_x, eer_y+0.0125, f'({eer_x}, {eer_y})', fontsize=14, color='black', ha='center')
title = f"{'MSE, IR='} {IR}"
plt.title(title)

plt.xlabel('K',fontsize=12)
plt.ylabel('Percentage',fontsize=12)
plt.legend(fontsize=12)
plt.show()

#%% plot CS
#plt FNIR,FPIR
x =np.arange(0,10,0.5)
plt.figure()
plt.plot(x,fnir_all_CS,color='#800080',label= 'FNIR')
plt.plot(x,fpir_all_CS,color='#6A5ACD',label= 'FPIR')

#plt EER
idx = 0
for i in range(len(x)):
    if fnir_all_CS[i] < fpir_all_CS[i]:
        idx = i
        break;
x1 = idx*0.5
x2 = (idx-1)*0.5
y1 = fnir_all_CS[idx]
y2 = fnir_all_CS[idx-1]
y3 = fpir_all_CS[idx]
y4 = fpir_all_CS[idx-1]
eer_x = ((x1*y2-y1*x2)*(x1-x2)-(x1-x2)*(x1*y4-y3*x2))/((x1-x2)*(y3-y4)-(y1-y2)*(x1-x2))
eer_y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x1*y4-y3*x2))/((x1-x2)*(y3-y4)-(y1-y2)*(x1-x2))
eer_x = round(eer_x,4)
eer_y = round(eer_y,4)

#eer_x = round(idx*0.5,4)
#eer_y = round(fnir_all[idx],4)

plt.plot(eer_x, eer_y, marker='o', markersize=8, color='#6495ed', label='EER')
plt.text(eer_x, eer_y+0.0125, f'({eer_x}, {eer_y})', fontsize=14, color='black', ha='center')
title = f"{'CS, IR='} {IR}"
plt.title(title)

plt.xlabel('K',fontsize=12)
plt.ylabel('Percentage',fontsize=12)
plt.legend(fontsize=12)
plt.show()
# #%%

# x1 = np.arange(640)

# for i in range(0,4700,20): #每人一個
#     y1 = template_test[i,:]
#     plt.plot(x1,y1,color ='g',alpha=0.2)
#     plt.title('Test Template Per Person')
# plt.show()

# for i in range(100,120,20): #一人全部
#     y1 = template_test[i,:]
#     plt.plot(x1,y1,color ='g',alpha=0.5)
#     plt.title('All Test Template for a person')
# plt.show()