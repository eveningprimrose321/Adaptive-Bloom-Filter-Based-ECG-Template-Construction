# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:19:40 2023

@author: ess601
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
from itertools import combinations
import matplotlib.pyplot as plt



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
    try:
        xdiff = (x1 - x2, x3 - x4)
        ydiff = (y1 - y2, y3 - y4)
    
        div = det(xdiff, ydiff)
        d = (det((x1,y1),(x2,y2)), det((x3,y3),(x4,y4)))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x,y
    except:
        return -1,-1

#%%
def bloom_filter(data ,feature_bit, parts):
    data_len = len(data)
    to_binary = np.vectorize(lambda x: bin(x)[2:].zfill(feature_bit*2))
    template = np.zeros((parts,2**feature_bit))
    for part in range(parts):
        datax = data[(data_len//parts)*part:(data_len//parts)*(part+1)].astype(int)
        data_bin = to_binary(datax.astype(int))
        data_bin = ''.join(data_bin)
        start_point = 0
        while start_point<feature_bit*(data_len//parts):
            temp = data_bin[start_point : start_point + feature_bit]
            template[part, int(temp, 2)] += 1
            start_point += feature_bit
    return np.reshape(template,(-1))

#%%
#template construction

file = loadmat('../../beatbundle_PTB_p_285.mat')
beatbundle = file['beatbundle']
data = np.zeros((285,50,1000))
for person in range (285):
    for beats in range (50):
        data[person,beats,:] = beatbundle[person, beats].flatten()
        
data_train = data[:,:30,:]
data_test = data[:,30:,:]
data_max = np.max(data_train)
data_min = np.min(data_train)


eer_rec = -1*np.ones((4,6,3))
for rec_i,feature_bit in enumerate([5,6,7,8]):
    ## feature_bit: trun decimal numbers into how many bits of binary
    ## (and it will be resized to double of it later)
    ## normalize the data into the range of (0,2**(2*feature_bit))
    ## 1.1 is to ensure that the test data will be within (0,2**(2*feature_bit))
    data_train = (data_train - data_min)*(2**(2*feature_bit))/((data_max - data_min)*1.1)
    data_test = (data_test - data_min)*(2**(2*feature_bit))/((data_max - data_min)*1.1)
    data_test = np.clip(data_test, 0, 2**(2*feature_bit))
    for rec_j,parts in enumerate([5,10,20,25,40,50]):
        for trial in range(3):
            template = []
            label = []
            avg_idx = np.random.randint(0, 15, (60,3))
            for p in range(235):
                template_per_person = []
                for i,idx in enumerate(avg_idx):
                    temp_ecg = np.mean(data_train[p,idx,:],axis=0)
                    template_per_person.append(bloom_filter(temp_ecg,feature_bit,parts))
                template.append(np.mean(np.array(template_per_person),axis = 0))
                label.append(p)
                                
            template = np.array(template)
            label = np.array(label)
                
            from sklearn.neighbors import KNeighborsClassifier 
            
            Classifier = KNeighborsClassifier(n_neighbors=1)
            Classifier.fit(template, label)
            
            #%%
            #iqr measurement
            template_iqr = np.zeros((235,100,(2**feature_bit)*parts))
            mse_all = np.zeros((235,100))
            avg_idx = np.random.randint(15, 30, (100,5))
            for p in range(235):
                for i,idx in enumerate(avg_idx):
                    temp_ecg = np.mean(data_train[p,idx,:],axis=0)
                    template_iqr[p,i,:] = bloom_filter(temp_ecg,feature_bit,parts)
                    mse_all[p,i] = np.mean((template_iqr[p,i,:]-template[p,:])**2)
                    
            iqr = np.zeros((235,3)) #q1,q3,iqr
            for p in range(235):
                iqr[p,0], iqr[p,1] = np.percentile(mse_all[p,:],[25,75])
                iqr[p,2] = iqr[p,1] - iqr[p,0]
                
            #%%
            #enrollee idenfication
            template_test = []
            label_test = []
            avg_idx = np.random.randint(0, 20, (20,5))
            for p in range(235):
                for i,idx in enumerate(avg_idx):
                    temp_ecg = np.mean(data_test[p,idx,:],axis=0)
                    template_test.append(bloom_filter(temp_ecg,feature_bit,parts))
                    label_test.append(p)
            template_test = np.array(template_test)
            predict_test = Classifier.predict(template_test)
            IR = sum(predict_test==np.array(label_test))/len(predict_test)
            
            #%%
            #outlier idenfication
            template_out = []
            for p in range(50):
                for i,idx in enumerate(avg_idx):
                    temp_ecg = np.mean(data_test[235+p,idx,:],axis=0)
                    template_out.append(bloom_filter(temp_ecg,feature_bit,parts))
            template_out = np.array(template_out)
            predict_out = Classifier.predict(template_out)
            
            #%%
            #verification
            fpir = []
            fnir = []
            k_range = np.arange(-2,10,0.5)
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
            thre, eer = intersection(fpir,fnir,k_range)
            eer_rec[rec_i,rec_j,trial] = eer
            # if trial == 1:
            #     fig = plt.figure(facecolor="w", figsize=(10, 5))
            #     plt.plot(k_range,fpir)
            #     plt.plot(k_range,fnir)
            #     plt.scatter(thre,eer)
            #     plt.annotate(f' EER = {eer*100:.2f}%', (thre, eer),xytext = (1, 1),textcoords='offset points')
            #     plt.title(f'FPIR/FNIR curve,system 1,feature_bit={feature_bit},parts = {parts}')
            #     plt.legend(["FPIR", "FNIR"])
            #     plt.ylabel("(%)")
            #     plt.xlabel("k")
            #     plt.grid()
            #     fig.savefig(f'./temp_fig/system_1/plt{feature_bit}_{parts}_{trial}.png') 
            #     plt.show()
            

# np.save('system_1_record', eer_rec)