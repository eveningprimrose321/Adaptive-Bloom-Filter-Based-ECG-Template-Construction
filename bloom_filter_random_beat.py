# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:19:40 2023

@author: ess601
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
from itertools import combinations
# file = loadmat('beatbundle_PTB_p_285.mat')
# beatbundle = file['beatbundle']

data_ori = np.load('./ptb_data/random_cut/x_train_1000.npy')

#%%
data = np.zeros((285,150,500))
for person in range (285):
    for beats in range (150):
        data[person,beats,:] = decimate(data_ori[person*150+beats,:].flatten(), 2)
data_train = data[:235,:100,:]
data_test = data[:,100:,:]
data_max = np.max(data_train)
data_min = np.min(data_train)
data_train = (data_train - data_min)*(2**11)/(data_max - data_min)
data_test = (data_test - data_min)*(2**11)/(data_max - data_min)
    
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]
def intersection(array1,array2,x_tick):
    if array1[0]>array2[0]:
        for i in range(1,len(array1)):
            if array1[i]<array2[i]:
                x1,y1 = x_tick[i-1],array1[i-1]
                x2,y2 = x_tick[i],array1[i]
                x3,y3 = x_tick[i-1],array2[i-1]
                x4,y4 = x_tick[i],array2[i]
                break
            elif array1[i]==array2[i]:return x_tick[i],array1[i]
    else:
        for i in range(1,len(array1)):
            if array1[i]>array2[i]:
                x1,y1 = x_tick[i-1],array1[i-1]
                x2,y2 = x_tick[i],array1[i]
                x3,y3 = x_tick[i-1],array2[i-1]
                x4,y4 = x_tick[i],array2[i]
                break
            elif array1[i]==array2[i]:return x_tick[i],array1[i]
    xdiff = (x1 - x2, x3 - x4)
    ydiff = (y1 - y2, y3 - y4)

    div = det(xdiff, ydiff)
    d = (det((x1,y1),(x2,y2)), det((x3,y3),(x4,y4)))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x,y

#%%

def bloom_filter(data,feature_bit):
    to_binary = np.vectorize(lambda x: bin(x)[2:])
    template = np.zeros((5,2**feature_bit))
    for part in range(5):
        datax = data[100*part:100*(part+1)].astype(int)
        datax_bin = to_binary(datax)
        feature = np.zeros((100,feature_bit*2))
        for bit in range (100):
            string = datax_bin[bit]
            char_list = [char for char in string]
            a = len(char_list)
            b = feature_bit*2-a
            feature[bit,b:] = char_list
        feature = np.reshape(feature,(200,feature_bit))
        for row in range(200):
            d_0 = 0
            for bit in range(feature_bit):
                temp = (2**(feature_bit-1-bit))*feature[row,bit]
                d_0 += temp
            template[part,int(d_0)] += 1
    return np.reshape(template,(-1))


#%%
#template construction


template = []
label = []
for p in range(235):
    template_per_person = []
    for i in range(50):
        template_per_person.append(bloom_filter(data_train[p, i, :],6))
    template.append(np.mean(np.array(template_per_person),axis = 0))
    label.append(p)
                    
template = np.array(template)
label = np.array(label)
    
from sklearn.neighbors import KNeighborsClassifier 

Classifier = KNeighborsClassifier(n_neighbors=1)
Classifier.fit(template, label)

#%%
#iqr measurement
template_iqr = np.zeros((235,50,320))
mse_all = np.zeros((235,50))
for p in range(235):
    for i,idx in enumerate(range(50,100)):
        template_iqr[p,i,:] = bloom_filter(data_train[p,idx,:],6)
        mse_all[p,i] = np.mean((template_iqr[p,i,:]-template[p,:])**2)
        
iqr = np.zeros((235,3)) #q1,q3,iqr
for p in range(235):
    iqr[p,0], iqr[p,1] = np.percentile(mse_all[p,:],[25,75])
    iqr[p,2] = iqr[p,1] - iqr[p,0]
    
#%%
#enrollee idenfication
template_test = []
label_test = []
for p in range(235):
    for idx in range(50):
        template_test.append(bloom_filter(data_test[p,idx,:],6))
        label_test.append(p)
template_test = np.array(template_test)
predict_test = Classifier.predict(template_test)
IR = sum(predict_test==np.array(label_test))/len(predict_test)

#%%
#outlier idenfication
template_out = []
for p in range(50):
    for idx in range(50):
        template_out.append(bloom_filter(data_test[p,idx,:],6))
template_out = np.array(template_out)
predict_out = Classifier.predict(template_out)

#%%
#verification
fpir = []
fnir = []
k_range = np.arange(0,10,0.5)
for k in k_range:
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
    fpir.append(fp/len(predict_out))
    fnir.append(fn/len(predict_test))


#%% plot
import matplotlib.pyplot as plt
# thre, eer = intersection(fpir,fnir,k_range)
# eer_rec.append(eer)
# thre_rec.append(thre)
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(k_range,fpir)
plt.plot(k_range,fnir)
# plt.scatter(thre,eer)
# plt.annotate(f' EER = {eer:.2f}%', (thre, eer),xytext = (1, 1),textcoords='offset points')
plt.title("FPIR/FNIR curve")
plt.legend(["FPIR", "FNIR"])
plt.ylabel("(%)")
plt.xlabel("k")
plt.grid()
# fig.savefig(f'./temp_fig/plt{trial}.png') 
plt.show()


