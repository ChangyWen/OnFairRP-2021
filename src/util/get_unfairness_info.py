#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import pandas as pd
from utils import save_obj
import os
from ctypes import c_wchar_p

def cal_fairness(k):
    discounts_ = np.frombuffer(discounts.get_obj())
    taus_ = np.frombuffer(taus.get_obj())
    ods_ = np.vstack([np.frombuffer(olats.get_obj()), np.frombuffer(olngs.get_obj()),
                      np.frombuffer(dlats.get_obj()), np.frombuffer(dlngs.get_obj())]).transpose()
    od_sq_sum_ = np.frombuffer(od_sq_sum.get_obj())
    fair = []
    od_weight = 16.9071  # TODO: od_weight, 15-->16.9071
    start = 1500 * k
    end = start + 1500 if k < m.value // 1500 else m.value

    for i in range(start, end):
        od_dis = od_sq_sum_[i] + od_sq_sum_[:i] - 2 * np.sum(np.multiply(ods_[i], ods_[:i]), axis=1)
        tau_dis = np.not_equal(taus_[i], taus_[:i]).astype(int)
        sim = od_weight * od_dis + tau_dis
        fair.append(np.where(K.value * sim + discounts_[:i] < discounts_[i])[0].tolist())

    save_obj(fair, directory.value + '/{}.pkl'.format(k))
    # save_obj(fair, '../data/unfair_linear_K{}_wk2/{}.pkl'.format(K.value, k)) #todo: filename!!!!
    return

def init(directory_, K_, m_, discounts_, taus_, olats_, olngs_, dlats_, dlngs_, od_sq_sum_):
    global directory, K, m, discounts, taus, olats, olngs, dlats, dlngs, od_sq_sum
    directory = directory_
    K = K_
    m = m_
    discounts = discounts_
    taus = taus_
    olats, olngs, dlats, dlngs = olats_, olngs_, dlats_, dlngs_
    od_sq_sum = od_sq_sum_

def get_unfair_list(K_:float, discount_col:str, directory_:str, all_riders:pd.DataFrame) -> np.array:
    # all_riders = pd.read_csv(
    #     '../data/__wkday_in_wk{}_h3.csv'.format(2), sep=',', header='infer',
    #     usecols=['olat', 'olng', 'dlat', 'dlng', 'tau', discount_col],
    #     dtype={'tau': int, 'olat': float, 'olng': float, 'dlat': float,
    #            'dlng': float, discount_col: float}
    # )
    os.makedirs(directory_, exist_ok=True)

    directory = mp.Value(c_wchar_p, directory_)
    K = mp.Value('d', K_)
    m = mp.Value('i', len(all_riders.index))
    discounts = mp.Array('d', all_riders[discount_col].to_numpy().tolist())
    taus = mp.Array('d', all_riders['tau'].to_numpy().astype(int).tolist())
    olats = mp.Array('d', all_riders['olat'].to_numpy().tolist())
    olngs = mp.Array('d', all_riders['olng'].to_numpy().tolist())
    dlats = mp.Array('d', all_riders['dlat'].to_numpy().tolist())
    dlngs = mp.Array('d', all_riders['dlng'].to_numpy().tolist())
    ods = all_riders[['olat', 'olng', 'dlat', 'dlng']].to_numpy()
    od_sq_sum = mp.Array('d', np.sum(np.multiply(ods, ods), axis=1).tolist())
    del all_riders, ods

    pool = mp.Pool(processes=4, initializer=init, initargs=(directory, K, m, discounts, taus, olats, olngs, dlats, dlngs, od_sq_sum))
    iters = range(m.value // 1500 + 1)
    pool.map(func=cal_fairness, iterable=iters)
    return

# if __name__ == '__main__':
#     get_unfair_list(0.32, 'ProfitOnly_exp_p0.05_r1.0', '../../data/uf_ProfitOnly_exp_p0.05_r1.0_K0.32_wk2')
    # get_unfair_list(0.04)
    # print('0.04 done')
    # time.sleep(18)
    # get_unfair_list(0.08)
    # print('0.08 done')
    # time.sleep(18)
    # get_unfair_list(0.64)
    # print('0.64 done')
    # time.sleep(18)
    # get_unfair_list(1.28)
    # print('1.28 done')
    # time.sleep(18)
    # get_unfair_list(2.56)
    # print('2.56 done')
    # time.sleep(18)
    # get_unfair_list(5.12)
    # print('5.12 done')
    # get_unfair_list(0.0)
    # print('0.0 done')