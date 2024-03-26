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
file = loadmat('../../ptb_data/beatbundle_PTB_p_285.mat')
beatbundle = file['beatbundle']
# 
# data_ori = np.load('./ptb_data/random_cut/x_train_1000.npy')

#%%

    

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

def bloom_filter(data , feature_bit, parts, key):
    if len(key) != parts:
        key = [key for i in range(parts)]
    data_len = len(data)
    to_binary = np.vectorize(lambda x: bin(x)[2:].zfill(feature_bit*2))
    template = np.zeros((parts,2**feature_bit))
    for part in range(parts):
        datax = data[(data_len//parts)*part:(data_len//parts)*(part+1)].astype(int)
        data_bin = to_binary(datax.astype(int))
        data_bin = ''.join(data_bin)
        start_point = 0
        for kth_key in key[part]:
            temp = data_bin[start_point : start_point + kth_key]
            template[part, int(temp, 2)] += 1
            start_point += kth_key
            if kth_key == key[-1]:
                temp = data_bin[start_point:]
                template[int(temp, 2)] += 1
                break
    return np.reshape(template,(-1))
            
    


#%%
eer_rec = np.ones((4,6,3))*-1
for rec_i,feature_bit in enumerate([5,6,7,8]):
    data = np.zeros((285,50,1000))
    for person in range (285):
        for beats in range (50):
            data[person,beats,:] = beatbundle[person, beats].flatten()
    data_train = data[:,:30,:]
    data_test = data[:,30:,:]
    data_max = np.max(data_train)
    data_min = np.min(data_train)
    data_train = (data_train - data_min)*(2**(2*feature_bit))/((data_max - data_min)*1.1)
    data_test = (data_test - data_min)*(2**(2*feature_bit))/((data_max - data_min)*1.1)
    for rec_j,parts in enumerate([5,10,20,25,40,50]):
        for trial in range(3):
            # print(trial)
            ### different key per part
            data_len = len(data_train[0,0,:])
            key = []
            for person in range(285):
                key_person = []
                for part in range(parts):
                    key_person_part = []
                    while True:
                        key_person_part.append(np.random.randint(1, feature_bit+1))
                        if sum(key_person_part) > (data_len//parts)*2*feature_bit:
                            key_person_part.pop()
                            key_person.append(key_person_part)
                            break
                key.append(key_person)
                
                
                
            # ### same key per part
            # key = []
            # for person in range(285):
            #     key_person = []
            #     while True:
            #         key_person.append(np.random.randint(4, 7))
            #         if sum(key_person)>100*12:
            #             key_person.pop()
            #             break
            #     key.append(key_person)
                    
            #%%
            ## template construction
            ## 有做平均template
            avg_idx = np.random.randint(0, 15, (60,3))
            template = []
            label = []
            
            for p in range(285):
                template_per_person = []
                for idx in avg_idx:
                    temp_ecg = np.mean(data_train[p,idx,:],axis=0)
                    template_per_person.append(bloom_filter(temp_ecg, feature_bit, parts, key[p]))
                template.append(np.mean(np.array(template_per_person),axis = 0))
                label.append(p)
                
                
            # ### 沒做平均template
            # # avg_idx = list(combinations([1, 2, 3, 4, 0], 3))
            # template = []
            # label = []
            
            # for p in range(285):
            #     template_per_person = []
            #     temp_ecg = np.mean(data_train[p,:3,:],axis=0)
            #     # for idx in avg_idx:
            #     #     temp_ecg = np.mean(data_train[p,idx,:],axis=0)
            #     #     template_per_person.append(bloom_filter(temp_ecg,feature_bit,key[p]))
            #     template.append(bloom_filter(temp_ecg,feature_bit,key[p]))
            #     label.append(p)
                
            template = np.array(template)
            label = np.array(label)
            template_len = len(template[0,:])
            
            #%%
            ### iqr measurement
            avg_idx = np.random.randint(15, 30, (100,5))
            template_iqr = np.zeros((285,100,template_len))
            mse_all = np.zeros((285,100))
            
            for p in range(285):
                for i,idx in enumerate(avg_idx):
                    temp_ecg = np.mean(data_train[p,idx,:],axis=0)
                    template_iqr[p,i,:] = bloom_filter(temp_ecg, feature_bit, parts, key[p])
                    mse_all[p,i] = np.mean((template_iqr[p,i,:]-template[p,:])**2)
            
            
            iqr = np.zeros((285,3)) #q1,q3,iqr
            iqr[:,0], iqr[:,1] = np.percentile(mse_all,[25,75],axis=1)
            iqr[:,2] = iqr[:,1] - iqr[:,0]
            
             
            #%%
            #verification
            test_time = 3
            far = []
            frr = []
            k_range = np.arange(0,10,0.5)
            
            avg_idx = np.random.randint(0, 20, (test_time,5))
            mse_test = np.zeros((285,285,test_time))
            
            for person in range(285):
                for goal_id in range(285):
                    for i, idx in enumerate(avg_idx):
                        temp_ecg = np.mean(data_test[person,idx,:],axis=0)
                        temp_template = bloom_filter(temp_ecg, feature_bit, parts, key[goal_id])
                        mse_test[person, goal_id, i] = np.mean((temp_template - template[goal_id,:])**2)
                        
            #%%
            for k in k_range:
                threshold = iqr[:,1] + k*iqr[:,2]
                fr, fa = 0, 0
            
                # false negative
                for person in range(285):
                    fr += sum(mse_test[person, person, :] > threshold[person])
                
                for person in range(285):
                    for goal_id in range(285):
                        if person != goal_id:
                            fa += sum(mse_test[person, goal_id, :] < threshold[goal_id])
                
                far.append(fa/(285*284*test_time))
                frr.append(fr/(285*1*test_time))
            
            
            #%% plot
            thre, eer = intersection(far,frr,k_range)
            eer_rec[rec_i,rec_j,trial] = eer
            if trial == 1:
                fig = plt.figure(facecolor="w", figsize=(10, 5))
                plt.plot(k_range,far)
                plt.plot(k_range,frr)
                plt.scatter(thre,eer)
                plt.annotate(f' EER = {eer*100:.2f}%', (thre, eer),xytext = (1, 1),textcoords='offset points')
                plt.title(f'FAR/FRR curve,system 1,feature_bit={feature_bit},parts = {parts}')
                plt.legend(["FAR", "FRR"])
                plt.ylabel("(%)")
                plt.xlabel("k")
                plt.grid()
                fig.savefig(f'./temp_fig/system_2/plt_diffkey_{feature_bit}_{parts}.png') 
                plt.show()
            

np.save('system_2_record_diffkey', eer_rec)