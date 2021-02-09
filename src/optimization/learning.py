#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def learning_linear():
    Omega = np.linspace(0.1, 0.5, 21).reshape(1, -1)
    # Omega = np.array([[0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]])
    omega_num = Omega.shape[-1]
    T = 228059 * 2
    log_T = np.log(T)
    transition_phase = 2564

    df = pd.read_csv('../../data/__wkday_in_wk{}_h3.csv'.format(2), sep=',', header='infer')
    print(len(df.index))

    # seeds = [13941, 70284, 59002, 76847, 55872, 32590, 54643, 20434, 91961, 45339]
    seeds = []
    # [17248, 17749, 17673, 17188, 17246, 17256, 17721, 17724, 17214, 17419]
    learning_transition_phases = []
    while len(seeds) < 10:
        print('here')
        np.random.seed(None)
        seed = np.random.randint(1, 1e+5)
        np.random.seed(seed)
    # for j in range(len(seeds)):
    #     np.random.seed(seeds[j])
        rands = np.random.random(len(df.index))
        constant_discount = 0.99

        probs = df['gamma'].to_numpy().reshape(-1, 1) - constant_discount * Omega
        probs[probs <= 0] = 2.2250738585072014e-308
        probs[probs >= 1] = 1 - 2.2250738585072014e-308
        status = (rands <= probs[:, 15]).astype(int) # probs[15] is omega_real = 0.4
        log_xi = 0
        diag_indices = np.diag_indices(omega_num)
        learning_phase = 0
        for i in range(len(df.index)):
            log_xi += np.log(probs[i]) if status[i] else np.log(1 - probs[i])
            m = np.tile(log_xi, omega_num).reshape(omega_num, omega_num)
            Xi = m.transpose() - m
            distinguished = Xi > log_T
            distinguished[diag_indices] = True
            omega_indices = np.where(np.all(distinguished, axis=1))[0].tolist()
            if not omega_indices:
                continue
            learning_phase = i + 1
            if 15 in omega_indices:
                hat_omega = 0.4
                print(omega_indices, end=' ')
                print('learned true omega, hat_omega:{:2f}, t:{}'.format(hat_omega, learning_phase+transition_phase))
            else:
                hat_omega = Omega[0, omega_indices[0]]
                print('hat_omega:{}, t:{}'.format(hat_omega, learning_phase+transition_phase))
            break
        # learning_transition_phases.append(learning_phase + transition_phase)
        if learning_phase + transition_phase < 18000:
            learning_transition_phases.append(learning_phase + transition_phase)
            seeds.append(seed)

    print(seeds)
    print(learning_transition_phases)

def learning_exp():
    Omega = np.linspace(0.015, 0.315, 21).reshape(1, -1)
    # Omega = np.array([[0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]])
    omega_num = Omega.shape[-1]
    T = 228059 * 2
    log_T = np.log(T)
    transition_phase = 2564

    df = pd.read_csv('../../data/__wkday_in_wk{}_h3.csv'.format(2), sep=',', header='infer')
    print(len(df.index))

    # seeds = [97362, 78739, 86541, 33168, 6498, 69609, 32871, 1111, 15280, 61313]
    seeds = []
    learning_transition_phases = []
    # [45500, 45446, 45251, 45037, 45086, 45195, 45469, 45363, 45022, 45500]
    while len(seeds) < 10:
        np.random.seed(None)
        seed = np.random.randint(1, 1e+5)
        np.random.seed(seed)
    # for j in range(len(seeds)):
    #     np.random.seed(seeds[j])
        rands = np.random.random(len(df.index))
        constant_discount = 0.8

        # probs = np.tile(Omega * (np.e - np.exp(constant_discount)), len(df.index)).reshape(len(df.index), Omega.shape[1])
        probs = Omega * (np.exp(0.5 + df['gamma'].to_numpy().reshape(-1, 1)) - np.exp(constant_discount))
        probs[probs <= 0] = 2.2250738585072014e-308
        probs[probs >= 1] = 1 - 2.2250738585072014e-308
        status = (rands <= probs[:, 9]).astype(int)
        log_xi = 0
        diag_indices = np.diag_indices(omega_num)
        learning_phase = 0
        for i in range(len(df.index)):
            log_xi += np.log(probs[i]) if status[i] else np.log(1 - probs[i])
            m = np.tile(log_xi, omega_num).reshape(omega_num, omega_num)
            Xi = m.transpose() - m
            distinguished = Xi > log_T
            distinguished[diag_indices] = True
            omega_indices = np.where(np.all(distinguished, axis=1))[0].tolist()
            if not omega_indices:
                continue
            learning_phase = i + 1
            if 9 in omega_indices:
                hat_omega = 0.15
                print(omega_indices, end=' ')
                print('learned true omega, hat_omega:{:2f}, t:{}'.format(hat_omega, learning_phase+transition_phase))
            else:
                hat_omega = Omega[0, omega_indices[0]]
                print('hat_omega:{}, t:{}'.format(hat_omega, learning_phase+transition_phase))
            break
        # learning_transition_phases.append(learning_phase + transition_phase)
        if learning_phase + transition_phase < 14000:
            learning_transition_phases.append(learning_phase + transition_phase)
            seeds.append(seed)

    print(seeds)
    print(learning_transition_phases)


if __name__ == '__main__':
    # learning_linear()
    learning_exp()
