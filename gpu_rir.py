# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:00:27 2022

@author: 肖永雄
"""

import os
from tqdm import tqdm
import numpy as np
import pyroomacoustics as pra
import gpuRIR
import pandas as  pd
import random
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

train_data_csv = pd.read_csv("/root/data/zhvoice_csv/zhvoice_rir_mix_45.csv", sep=',').values
train_data = []
for t1, t2, p1, p2, m1, m2, sinr, label in train_data_csv:
    train_data.append([t1, t2, p1, p2, m1, m2, sinr, label])

M = 6
rmic = 0.035
T0 = 3
FS = 16000
in_wave_len = int(T0*FS)
out_wave_len = int(3*FS)
room_dim = np.array([40, 50, 8])  # meters

def mix_wrapper(num):
    _, _,phi1,phi2,mic_centerx, mic_centery, _, _ = train_data[num]
    room_mix(phi1,phi2,mic_centerx, mic_centery, num)

def room_mix(phi1,phi2,mic_centerx, mic_centery,num):
    deta_phi = random.randint(0,15)
    sig = random.choice((-1, 1))
    phi1 = int(phi1 + sig*deta_phi)
    rt60 = round(random.uniform(1.00,1.80),2)
    distance = round(random.uniform(1.00,1.40),2)
    x_s1 = mic_centerx+distance*np.cos(np.deg2rad(phi1))
    y_s1 = mic_centery+distance*np.sin(np.deg2rad(phi1))
    x_s2 = mic_centerx+distance*np.cos(np.deg2rad(phi2))
    y_s2 = mic_centery+distance*np.sin(np.deg2rad(phi2))
    spk1_pos = np.array([x_s1, y_s1, 1.70])
    spk2_pos = np.array([x_s2, y_s2, 1.70])
    spk_pos = (np.c_[spk1_pos,spk2_pos]).T
    # spk_pos = (spk2_pos).T

    R_2 = pra.circular_2D_array(center=[mic_centerx, mic_centery], M = M, phi0=0, radius=rmic)
    R_cir = np.concatenate((R_2, np.ones((1, 6)) * 1.68))
    mic_pos = R_cir.T
    
    beta = gpuRIR.beta_SabineEstimation(room_dim, rt60)
    nb_img = gpuRIR.t2n(rt60, room_dim)
    spk_rir = gpuRIR.simulateRIR(room_dim, beta, spk_pos, mic_pos, nb_img, rt60, FS)
    np.savez('/root/data/rir_gpu/gpurir_'+str(num) + '.npz', spk_rir)
            
if __name__ == '__main__':
    mp.set_start_method("spawn")
    cpu_num = 32
    train_idx = list(range(0,30000))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(mix_wrapper, train_idx), total=len(train_idx)))

    