#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from util.utils import calculate_similarity
from itertools import chain
from datetime import datetime
import os

'''global variable'''
filename:str
Nfeval = 1 # TODO: intermediate res, Nfeval = 1
dist:np.array
gamma:np.array
omega:float
price:np.array
cost:np.array
objs:list
oc_plus_gp:np.array # omega * cost + gamma * price
op:np.array # omega * price
t_op:np.array # 2 * omega * price
oep:np.array # omega * np.e * price
oc:np.array # omega * cost
pp:np.array # unfair_penalty * price
eg:np.array # np.exp(gamma)
o_eg_p:np.array # omega * eg * price
unfair_penalty:float

'''OnFair'''
def callbackF_ProfitOBJ(d):
    global Nfeval, objs
    if Nfeval % 11 == 1:
        obj = ProfitOBJ(d)
        objs.append(obj)
        print('\n{0}:  {1:3d}  {2: 3.6f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), Nfeval, obj))
    np.savetxt('../output/' + filename + '~{}'.format(Nfeval), d)
    np.savetxt('../output/p-' + filename + '~{}'.format(Nfeval), np.array(objs))
    if os.path.exists('../output/' + filename + '~{}'.format(Nfeval - 1)):
        os.remove('../output/' + filename + '~{}'.format(Nfeval - 1))
    if os.path.exists('../output/p-' + filename + '~{}'.format(Nfeval - 1)):
        os.remove('../output/p-' + filename + '~{}'.format(Nfeval - 1))
    print('{0}~{1}'.format(datetime.now().strftime("%m/%d-%H:%M"), Nfeval), end=' ')
    Nfeval += 1
def callbackF_ProfitOBJ2(d):
    global Nfeval, objs
    # if Nfeval % 11 == 1:
    if True:
        obj = ProfitOBJ2(d)
        objs.append(obj)
        print('\n{0}:  {1:3d}  {2: 3.6f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), Nfeval, obj))
    np.savetxt('../output/' + filename + '~{}'.format(Nfeval), d)
    np.savetxt('../output/p-' + filename + '~{}'.format(Nfeval), np.array(objs))
    if os.path.exists('../output/' + filename + '~{}'.format(Nfeval - 1)):
        os.remove('../output/' + filename + '~{}'.format(Nfeval - 1))
    if os.path.exists('../output/p-' + filename + '~{}'.format(Nfeval - 1)):
        os.remove('../output/p-' + filename + '~{}'.format(Nfeval - 1))
    print('{0}~{1}'.format(datetime.now().strftime("%m/%d-%H:%M"), Nfeval), end=' ')
    Nfeval += 1
def ProfitOBJ(d):
    return -1 * dist.T.dot((gamma - d * omega) * (d * price - cost))
def grad_ProfitOBJ(d):
    return -1 * dist * (oc_plus_gp - t_op * d)
# def ProfitOBJ2(d):
#     return -1 * dist.T.dot((omega * (np.e - np.exp(d))) * (d * price - cost))
# def grad_ProfitOBJ2(d):
#     return -1 * dist * (oep - op * np.exp(d) * d - op * np.exp(d) + oc * np.exp(d))
def ProfitOBJ2(d):
    return -1 * dist.T.dot((omega * (eg - np.exp(d))) * (d * price - cost))
def grad_ProfitOBJ2(d):
    return -1 * dist * (o_eg_p - op * np.exp(d) * d - op * np.exp(d) + oc * np.exp(d))


def fair(distribution_file, week, omega_, isLinear, discount_a, K_similar_const):
    df = pd.read_csv(distribution_file)
    taus = df['tau'].to_numpy().astype(int)
    ods = df[['olat', 'olng', 'dlat', 'dlng']].to_numpy()
    similarity = calculate_similarity(ods, taus)
    odtau_num = len(df.index) # 1000

    global filename, dist, price, cost, gamma, omega, objs, oc_plus_gp, op, t_op, oep, oc, eg, o_eg_p
    objs = []
    omega = omega_
    dist = df.distribution.values[:odtau_num]
    price = df.price.values[:odtau_num]
    cost = df.cost.values[:odtau_num]
    gamma = df.gamma.values[:odtau_num]
    oc_plus_gp = omega * cost + gamma * price
    op = omega * price
    t_op = 2 * op
    oep = omega * np.e * price
    oc = omega * cost
    eg = np.exp(0.5 + gamma)
    o_eg_p = omega * eg * price

    colName = 'OnFair_linear' if isLinear else 'OnFair_exp'
    filename = '{}_K{}_o{}_a{}_wk{}'.format(colName, K_similar_const, omega, discount_a, week)
    m = odtau_num
    d0 = np.ones(m) * discount_a
    # d0 = np.loadtxt('../output/OnFair_linear_K5.12_o0.4_a0.99_wk1~19') # TODO: intermediate res
    tril_indices = np.tril_indices(m, -1)
    K_similarity_square = np.square(K_similar_const * similarity[tril_indices])
    indices1 = tril_indices[1]
    indices2 = np.array(list(chain.from_iterable([[i] * i for i in range(1, m)]))) # = tril_indices[0]
    del taus, ods, df, similarity, tril_indices

    constraints = []
    constraints.append({'type': 'ineq', 'fun': lambda d: d, 'jac': lambda d: np.identity(m)})  # lower bound = 0
    constraints.append({'type': 'ineq', 'fun': lambda d: discount_a - d, 'jac': lambda d: -np.identity(m)})  # upper bound = a
    def jacobian(d):
        g = np.zeros([len(indices1), m])
        for i in range(len(indices1)):
            g[i][indices2[i]] = -d[indices2[i]] + d[indices1[i]]
            g[i][indices1[i]] = d[indices2[i]] - d[indices1[i]]
        return g
    constraints.append({
        'type': 'ineq', 'fun': lambda d: 0.5 * (K_similarity_square - np.square(d[indices2] - d[indices1])), 'jac': jacobian
    })  # fairness constraints

    if isLinear:
        res = minimize(
            fun=ProfitOBJ, x0=d0, jac=grad_ProfitOBJ, method='SLSQP', callback=callbackF_ProfitOBJ,
            constraints=constraints, options={'maxiter': 12, 'disp': True, 'ftol':1e-6}, tol=1e-6
        ) # TODO: intermediate res 23 rounds
    else:
        res = minimize(
            fun=ProfitOBJ2, x0=d0, jac=grad_ProfitOBJ2, method='SLSQP', callback=callbackF_ProfitOBJ2,
            constraints=constraints, options={'maxiter': 12, 'disp': True, 'ftol':1e-6}, tol=1e-6
        )

    print(res.x[:10])
    np.savetxt('../output/' + filename + '.txt', res.x)
    np.savetxt('../output/p-' + filename + '.txt', np.array(objs))

###############################################################

'''Fixed'''
def callbackF_ProfitOBJ_Fixed(d):
    global Nfeval, objs
    obj = ProfitOBJ(d)
    objs.append(obj)
    print('{0}:  {1:3d}  {2:3.6f} {3:3.6f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), Nfeval, obj, d[0]))
    Nfeval += 1
def callbackF_ProfitOBJ2_Fixed(d):
    global Nfeval, objs
    obj = ProfitOBJ2(d)
    objs.append(obj)
    print('{0}:  {1:3d}  {2:3.6f}  {3:3.6f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), Nfeval, obj, d[0]))
    Nfeval += 1
def grad_ProfitOBJ_Fixed(d):
    return -1 * dist.T.dot(oc_plus_gp - t_op * d)
# def grad_ProfitOBJ2_Fixed(d):
#     return -1 * dist.T.dot(oep - op * np.exp(d) * d - op * np.exp(d) + oc * np.exp(d))
def grad_ProfitOBJ2_Fixed(d):
    return -1 * dist.T.dot(o_eg_p - op * np.exp(d) * d - op * np.exp(d) + oc * np.exp(d))

def fixed(distribution_file, week, omega_, isLinear, discount_a):
    df = pd.read_csv(distribution_file)

    global filename, dist, price, cost, gamma, omega, objs, oc_plus_gp, op, t_op, oep, oc, eg, o_eg_p
    objs = []
    omega = omega_
    dist = df.distribution.values
    price = df.price.values
    cost = df.cost.values
    gamma = df.gamma.values
    oc_plus_gp = omega * cost + gamma * price
    op = omega * price
    t_op = 2 * op
    oep = omega * np.e * price
    oc = omega * cost
    eg = np.exp(0.5 + gamma)
    o_eg_p = omega * eg * price

    d0 = discount_a
    colName = 'Fixed_linear' if isLinear else 'Fixed_exp'
    filename = '{}_o{}_a{}_wk{}.txt'.format(colName, omega, discount_a, week)
    m = 1
    print('Initial Solution: {}'.format(d0))
    constraints = []
    constraints.append({'type': 'ineq', 'fun': lambda d: d, 'jac': lambda d: np.identity(m)})  # lower bound = 0
    # constraints.append({'type': 'ineq', 'fun': lambda d: discount_a - d, 'jac': lambda d: -np.identity(m)})  # upper bound = a
    constraints.append({'type': 'ineq', 'fun': lambda d: 1 - d, 'jac': lambda d: -np.identity(m)})  # upper bound = a
    if isLinear == 1:
        res = minimize(fun=ProfitOBJ, x0=d0, jac=grad_ProfitOBJ_Fixed, method='SLSQP', callback=callbackF_ProfitOBJ_Fixed,
                   constraints=constraints, options={'maxiter': 12, 'disp': True, 'ftol':1e-6}, tol=1e-6)
    else:
        res = minimize(fun=ProfitOBJ2, x0=d0, jac=grad_ProfitOBJ2_Fixed, method='SLSQP', callback=callbackF_ProfitOBJ2_Fixed,
                   constraints=constraints, options={'maxiter': 12, 'disp': True, 'ftol':1e-6}, tol=1e-6)

    print(res.x[:10])
    np.savetxt('../output/' + filename, res.x)
    np.savetxt('../output/p-' + filename, np.array(objs))

###############################################################

'''ProfitOnly'''
def callbackF_ProfitOBJ_ProfitOnly(d):
    global Nfeval, objs
    obj = ProfitOBJ_ProfitOnly(d)
    objs.append(obj)
    print('{0}:  {1:3d}  {2:3.6f}  {3:3.6f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), Nfeval, obj, np.mean(d)))
    Nfeval += 1
def callbackF_ProfitOBJ2_ProfitOnly(d):
    global Nfeval, objs
    obj = ProfitOBJ2_ProfitOnly(d)
    objs.append(obj)
    print('{0}:  {1:3d}  {2:3.6f}  {3:3.6f}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), Nfeval, obj, np.mean(d)))
    Nfeval += 1
def ProfitOBJ_ProfitOnly(d):
    return -1 * dist.T.dot((gamma - d * omega - unfair_penalty) * (d * price - cost))
def grad_ProfitOBJ_ProfitOnly(d):
    return -1 * dist * (oc_plus_gp - t_op * d - pp)
# def ProfitOBJ2_ProfitOnly(d):
#     return -1 * dist.T.dot((omega * (np.e - np.exp(d)) - unfair_penalty) * (d * price - cost))
# def grad_ProfitOBJ2_ProfitOnly(d):
#     return -1 * dist * (oep - op * np.exp(d) * d - op * np.exp(d) + oc * np.exp(d) - pp)
def ProfitOBJ2_ProfitOnly(d):
    return -1 * dist.T.dot((omega * (eg - np.exp(d)) - unfair_penalty) * (d * price - cost))
def grad_ProfitOBJ2_ProfitOnly(d):
    return -1 * dist * (o_eg_p - op * np.exp(d) * d - op * np.exp(d) + oc * np.exp(d) - pp)

def profitOnly(distribution_file, week, omega_, isLinear, discount_a, penalty_, ratio_):
    df = pd.read_csv(distribution_file)

    global filename, dist, price, cost, gamma, omega, objs, oc_plus_gp, op, t_op, oep, oc, pp, unfair_penalty, eg, o_eg_p
    unfair_penalty = ratio_ * penalty_
    objs = []
    omega = omega_
    dist = df.distribution.values
    price = df.price.values
    cost = df.cost.values
    gamma = df.gamma.values
    oc_plus_gp = omega * cost + gamma * price
    op = omega * price
    t_op = 2 * op
    oep = omega * np.e * price
    oc = omega * cost
    pp = unfair_penalty * price
    eg = np.exp(0.5 + gamma)
    o_eg_p = omega * eg * price

    colName = 'ProfitOnly_linear' if isLinear else 'ProfitOnly_exp'
    filename = '{}_o{}_a{}_wk{}_p{}_r{}.txt'.format(colName, omega, discount_a, week, penalty_, ratio_)
    m = len(df.index)
    d0 = np.ones(m) * discount_a

    constraints = []
    constraints.append({'type': 'ineq', 'fun': lambda d: d, 'jac': lambda d: np.identity(m)})  # lower bound = 0
    # constraints.append({'type': 'ineq', 'fun': lambda d: discount_a - d, 'jac': lambda d: -np.identity(m)})  # upper bound = a
    constraints.append({'type': 'ineq', 'fun': lambda d: 1 - d, 'jac': lambda d: -np.identity(m)})  # upper bound = a
    if isLinear == 1:
        res = minimize(
            fun=ProfitOBJ_ProfitOnly, x0=d0, jac=grad_ProfitOBJ_ProfitOnly, method='SLSQP',
            callback=callbackF_ProfitOBJ_ProfitOnly, constraints=constraints,
            options={'maxiter': 12, 'disp': True, 'ftol': 1e-6}, tol=1e-6
        )
    else:
        res = minimize(
            fun=ProfitOBJ2_ProfitOnly, x0=d0, jac=grad_ProfitOBJ2_ProfitOnly, method='SLSQP',
            callback=callbackF_ProfitOBJ2_ProfitOnly, constraints=constraints,
            options={'maxiter': 12, 'disp': True, 'ftol': 1e-6}, tol=1e-6
        )

    print(res.x[:10])
    np.savetxt('../output/' + filename, res.x)
    np.savetxt('../output/p-' + filename, np.array(objs))
