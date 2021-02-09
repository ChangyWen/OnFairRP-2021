#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from math import e
from typing import Dict, Tuple
from collections import defaultdict
import time
import pickle


def get_shortest_path(G, route_cache, o, d):
    exists = False
    if (o, d) in route_cache.keys():
        route = route_cache[(o, d)]
        exists = True
    else:
        try:
            route = nx.shortest_path(G, o, d, weight='length')
            route_cache[(o, d)] = route
            exists = True
        except:
            try:
                route = nx.shortest_path(G, d, o, weight='length')
                route.reverse()
                route_cache[(o, d)] = route_cache[(d, o)] = route
            except:
                route = -1
    return route.copy(), exists

def get_shortest_path_length(G, dis_cache, o, d):
    exists = False
    if (o, d) in dis_cache.keys():
        dis = dis_cache[(o, d)]
        exists = True
    else:
        try:
            dis = nx.shortest_path_length(G, o, d, weight='length')
            dis_cache[(o, d)] = dis
            exists = True
        except:
            try:
                dis = nx.shortest_path_length(G, d, o, weight='length')
                dis_cache[(o, d)] = dis_cache[(d, o)] = dis
            except:
                dis = -1
    return dis, exists

def demand_func(gamma=None, discount=None, omega=None, isLinear=False):
    if isLinear:
        # gamma = np.random.uniform(low=0.52, high=0.68, size=1) if tau == 0 else np.random.uniform(low=0.72, high=0.88, size=1)
        return np.clip(-omega * discount + gamma, a_min=0, a_max=1) # TODO: <0, omega <= 0.52
    else:
        return np.clip(omega * (np.exp(0.5 + gamma) - np.exp(discount)), a_min=0, a_max=1)
        # return np.clip(omega*(e - e**discount), a_min=0, a_max=1) # TODO: >1, omega <= 0.58
    # TODO: Omega, 0.02, 20

# def init_data(discount_col, rider_loc, route_cache_loc, dis_cache_loc) -> (pd.DataFrame, Dict, Dict, int):
#     df = pd.read_csv(
#         rider_loc, sep=',', header='infer',
#         usecols=['timestamp', 'olat', 'olng', 'dlat', 'dlng', 'o', 'd', discount_col, 'gamma'],
#         dtype={'timestamp': int, 'o': int, 'd': int, 'olat':float, 'olng':float, 'dlat':float, 'dlng':float,
#                discount_col:float, 'gamma':float}
#     )
#     dis_cache = load_obj(dis_cache_loc)
#     route_cache = load_obj(route_cache_loc)
#     return (df, route_cache, dis_cache, len(df.index))

def init_data(route_cache_loc, dis_cache_loc) -> (Dict, Dict):
    dis_cache = load_obj(dis_cache_loc)
    route_cache = load_obj(route_cache_loc)
    return (route_cache, dis_cache)

def init_unfair_info(filename):
    return load_obj(filename)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

'''
ind = np.tril_indices(n, -1)
od_sim = od_similarity[ind]
16.90710227649637 = 1/ np.mean(od_sim)
'''

def calculate_similarity(ods, taus):
    def od_dis(m, n):
        matrix1 = np.matmul(m, m.transpose())
        sq_sum = np.sum(np.multiply(m, m), axis=1)
        matrix2 = np.tile(sq_sum, n).reshape(n, n)
        return np.sqrt(np.clip(matrix2 + matrix2.transpose() - 2 * matrix1, a_min=0, a_max=None))
    def tau_dis(m, n):
        matrix = np.tile(m, n).reshape(n, n)
        return np.not_equal(matrix, matrix.transpose()).astype(int)

    n = ods.shape[0]
    assert ods.shape == (n, 4) and taus.shape == (n,)
    od_similarity = od_dis(ods, n)
    tau_similarity = tau_dis(taus, n)
    od_weight = 16.9071 # TODO: od_weight, 15-->16.9071
    similarity = od_weight * od_similarity + tau_similarity
    return similarity

# TODO: weight of od-distance
# def get_unfair_matrix(K:float) -> np.array:
#     for week in [1,2,3]:
#         all_riders = pd.read_csv(
#             '../../data/_wkday_in_wk{}_h3.csv'.format(week), sep=',', header='infer',
#             usecols=['olat', 'olng', 'dlat', 'dlng', 'tau'],
#             dtype={'tau': int, 'olat': np.float32, 'olng': np.float32, 'dlat': np.float32,
#                    'dlng': np.float32}
#         )
#         n = all_riders.shape[0]
#         discounts = np.tile(all_riders['ProfitOnly'].to_numpy(), n).reshape(n, n).astype(np.float32)
#         taus = all_riders['tau'].to_numpy().astype(int)
#         ods = all_riders[['olat', 'olng', 'dlat', 'dlng']].to_numpy().astype(np.float32)
#
#         similarity = calculate_similarity(ods, taus)
#         unfair_matrix = np.tril(discounts.transpose() - discounts > K * similarity)
#         print(n, unfair_matrix.shape, unfair_matrix.dtype)
#         np.savetxt('../../data/unfair_matrix_wk{}_K{}.txt'.format(week, K), unfair_matrix)

# def get_unfair_list(K:float) -> np.array:
#     discount_col = 'ProfitOnly_linear'
#     all_riders = pd.read_csv(
#         '../../data/__wkday_in_wk{}_h3.csv'.format(2), sep=',', header='infer',
#         usecols=['olat', 'olng', 'dlat', 'dlng', 'tau', discount_col],
#         dtype={'tau': int, 'olat': float, 'olng': float, 'dlat': float,
#                'dlng': float, discount_col: float}
#     )
#
#     m = len(all_riders.index)
#     discounts = all_riders[discount_col].to_numpy()
#     taus = all_riders['tau'].to_numpy().astype(int)
#     ods = all_riders[['olat', 'olng', 'dlat', 'dlng']].to_numpy()
#     od_sq_sum = np.sum(np.multiply(ods, ods), axis=1)
#     del all_riders
#
#     fair = []
#     od_weight = 16.9071  # TODO: od_weight, 15-->16.9071
#     for i in range(1, m):
#         if i % 5000 == 0: print(i)
#         od_dis = od_sq_sum[i] + od_sq_sum[:i] - 2 * np.sum(np.multiply(ods[i], ods[:i]), axis=1)
#         tau_dis = np.not_equal(taus[i], taus[:i]).astype(int)
#         sim = od_weight * od_dis + tau_dis
#         # unfair.extend((K * sim + discounts[:i] < discounts[i]).tolist())
#         # fair[i] = set(np.where(K * sim + discounts[:i] >= discounts[i])[0].tolist())
#         set(np.where(K * sim + discounts[:i] >= discounts[i])[0].tolist())
#         # fair.append(np.where(K * sim + discounts[:i] >= discounts[i])[0].tolist())

    # save_obj(fair, '../../data/fair_K{}_wk2.pkl'.format(K))

# def get_unfair_dict(K:float) -> Dict:
#     for week in [1]:
#         all_riders = pd.read_csv(
#             '../../data/_wkday_in_wk{}_h3.csv'.format(week), sep=',', header='infer',
#             usecols=['olat', 'olng', 'dlat', 'dlng', 'tau'],
#             dtype={'tau': np.float32, 'olat': np.float32, 'olng': np.float32, 'dlat': np.float32,
#                    'dlng': np.float32}
#         ) #TODO: read discounts
#         n = all_riders.shape[0]
#         discounts = np.random.uniform(0, 1, n).astype(np.float32)
#         # discounts = all_riders['ProfitOnly'].to_numpy().astype(np.float32) # TODO: use this
#         taus = all_riders['tau'].to_numpy().astype(np.float32)
#         ods = all_riders[['olat', 'olng', 'dlat', 'dlng']].to_numpy().astype(np.float32)
#         del all_riders
#         unfair = []
#         unfair.append(set())
#         a = time.time()
#         for i in range(1, n):
#             if i % 10000 == 0:
#                 print(i)
#                 print(time.time() - a)
#                 a = time.time()
#             cur_od = ods[i]
#             cur_tau = taus[i]
#             # cur_unfair = discounts[i] > discounts[:i] + K * (np.linalg.norm(cur_od - ods[:i], axis=1) + cur_tau == taus[:i])
#             cur_unfair = discounts[i] > discounts[:i] + K * (np.sqrt(np.sum(np.square(cur_od - ods[:i]), axis=1)) + cur_tau == taus[:i])
#             unfair.append(set(np.where(cur_unfair)[0]))
#         save_obj(unfair, '../../data/unfair_wk{}_K{}.pkl'.format(week, K))
#         print(week)

def find_bbox():
    li = []
    for filename in ['../../data/wkday_in_wk1_h3.csv', '../../data/wkday_in_wk2_h3.csv', '../../data/wkday_in_wk3_h3.csv',
                     '../../data/wkday_in_wk1_sum_h3.csv', '../../data/wkday_in_wk2_sum_h3.csv', '../../data/wkday_in_wk3_sum_h3.csv']:
        df_ = pd.read_csv(
            filename, sep=',', header='infer',
            usecols=['olat', 'olng', 'dlat', 'dlng'],
            dtype={'olat':float, 'olng':float, 'dlat':float, 'dlng':float}
        )
        li.append(df_)
    df = pd.concat(li, axis=0, ignore_index=True)
    north = max(df['olat'].max(), df['dlat'].max())
    south = min(df['olat'].min(), df['dlat'].min())
    west = min(df['olng'].min(), df['dlng'].min())
    east = max(df['olng'].max(), df['dlng'].max())
    print(north, south, west, east)
    '''
    1600+ odtau:
    30.731459 30.650341 104.038645 104.13389
    '''
    '''
    1024 odtau:
    30.730364 30.650341 104.038645 104.13389
    '''

def complement_sum(output_file, discount_col):
    df = pd.read_csv('../data/_wkday_in_wk{}_sum_h3.csv'.format(1), sep=',', header='infer', usecols=['gamma', 'o', 'd', 'olat', 'olng', 'dlat', 'dlng', 'tau', 'ogrid_h3', 'dgrid_h3'])
    discounts = np.loadtxt('../output/'+output_file)
    df[discount_col] = discounts
    return df
    # df.to_csv('../data/_wkday_in_wk{}_sum_h3.csv'.format(1), sep=',', index=False)


def pre_data(sum_df):
    df = pd.read_csv(
        '../data/__wkday_in_wk{}_h3.csv'.format(2), sep=',', header='infer',
        usecols=['timestamp', 'tau', 'ogrid_h3', 'dgrid_h3', 'user_ids'],
        dtype={'timestamp': int, 'tau': int, 'ogrid_h3':str, 'dgrid_h3':str}
    )
    # sum_df = pd.read_csv('../data/_wkday_in_wk{}_sum_h3.csv'.format(1), sep=',', header='infer')
    # sum_df.drop(columns=['dist', 'price', 'cost', 'distribution'], inplace=True)
    res = pd.merge(df, sum_df, how='inner', on=['tau', 'ogrid_h3', 'dgrid_h3'])
    res.sort_values(by=['timestamp', 'ogrid_h3', 'dgrid_h3', 'gamma', 'o', 'd'], inplace=True)
    return res
    # res.to_csv('../data/__wkday_in_wk{}_h3.csv'.format(2), sep=',', index=False)


def pre_data_():
    # ox.config(log_console=False, use_cache=True)
    G = ox.graph_from_bbox(
        north=30.730364 + 0.02, south=30.650341 - 0.02,
        west=104.038645 - 0.02, east=104.13389 + 0.02,
        network_type='drive'
    )
    for week in [1, 2, 3]:
        df = pd.read_csv(
            '../../data/wkday_in_wk{}_sum_h3.csv'.format(week), sep=',', header='infer',
            # usecols=['olng', 'olat', 'dlat', 'dlng', 'distribution', 'gamma', 'tau'],
            dtype={'gamma': float, 'distribution': float, 'tau': int, 'olat':float, 'olng':float, 'dlat':float, 'dlng':float}
        )
        print(len(df.index))
        olngs = df['olng'].to_list()
        olats = df['olat'].to_list()
        dlngs = df['dlng'].to_list()
        dlats = df['dlat'].to_list()
        assert len(olngs) == len(olats) == len(dlngs) == len(dlats)
        lngs = olngs + dlngs
        lats = olats + dlats
        nodes = ox.get_nearest_nodes(G, lngs, lats, method='kdtree').tolist()
        assert len(nodes) == 2 * len(olngs)
        df['o'] = nodes[:len(olngs)]
        df['d'] = nodes[len(olngs):]
        fee_per_unit = 0.003 # TODO: fee_per_unit
        cost_per_unit = 0.003 * 0.25 # TODO: cost_per_unit
        route_dist = []
        dis_cache = {}
        for o_, d_ in zip(df['o'], df['d']):
            route_dist.append(get_shortest_path_length(G, dis_cache, o_, d_)[0])
        df['dist'] = route_dist
        df['price'] = fee_per_unit * df['dist']
        df['cost'] = cost_per_unit * df['dist']
        df.to_csv('../../data/_wkday_in_wk{}_sum_h3.csv'.format(week), sep=',', index=False)
        print('done-')

def get_cache():
    # ox.config(log_console=False, use_cache=True)
    G = ox.graph_from_bbox(
        north=30.730364 + 0.02, south=30.650341 - 0.02,
        west=104.038645 - 0.02, east=104.13389 + 0.02,
        network_type='drive'
    )
    dis_cache = {}
    route_cache = {}

    li = []
    for filename in [
        '../../data/_wkday_in_wk1_sum_h3.csv', '../../data/_wkday_in_wk2_sum_h3.csv', '../../data/_wkday_in_wk3_sum_h3.csv',
    ]:
        df_ = pd.read_csv(
            filename, sep=',', header='infer',
            usecols=['o', 'd'],
            dtype={'o': int, 'd': int}
        )
        li.append(df_)
    df = pd.concat(li, axis=0, ignore_index=True)

    print(len(df.index))

    i = 0
    a = time.time()

    for (o, d) in zip(df['o'], df['d']):
        if i % 10000 == 0:
            print(i, time.time() - a)
            print()
            a = time.time()
        i += 1
        if (o, d) not in dis_cache:
            dis_cache[(o, d)] = get_shortest_path_length(G, dis_cache, o, d)[0]
        if (o, d) not in route_cache:
            route_cache[(o, d)] = get_shortest_path(G, route_cache, o, d)[0]

    save_obj(dis_cache, '../../data/dis_cache.pkl')
    save_obj(route_cache, '../../data/route_cache.pkl')

def save_obj(d, filename):
    with open(filename, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # find_bbox()
    # get_cache()
    complement_sum()
    pre_data()
    # get_unfair_list(0.16)
    # get_unfair_dict(0.9)
    # pre_data_()
    # df = pd.read_csv(
    #     '../../data/wkday_in_wk{}_sum_h3.csv'.format(1), sep=',', header='infer',
    #     usecols=['olng', 'olat', 'dlat', 'dlng', 'distribution', 'gamma', 'tau'],
    #     dtype={'gamma': float, 'distribution': float, 'tau': int, 'olat': float, 'olng': float, 'dlat': float,
    #            'dlng': float}
    # )
    # taus = df['tau'].to_numpy().astype(int)
    # ods = df[['olat', 'olng', 'dlat', 'dlng']].to_numpy()
    # similarity = calculate_similarity(ods, taus)
    # df = pd.read_csv(
    #     '../../data/_wkday_in_wk{}_sum_h3.csv'.format(1), sep=',', header='infer',
    #     usecols=['olng', 'olat', 'dlat', 'dlng', 'distribution', 'gamma', 'tau'],
    #     dtype={'gamma': float, 'distribution': float, 'tau': int, 'olat': float, 'olng': float, 'dlat': float,
    #            'dlng': float}
    # )
    pass


    '''Usage of Osmnx'''
    '''
    ox.config(log_console=False, use_cache=True)
    G = ox.graph_from_address('Chengdu, China', distance=100000, network_type='drive')
    ox.plot_graph(G)
    graph_proj = ox.project_graph(G)
    nodes,edegs = ox.graph_to_gdfs(graph_proj)
    o = ox.get_nearest_node(G, (lat, lng))
    d = ox.get_nearest_node(G, (lat, lng))
    dis = nx.shortest_path_length(G, o, d, weight='length') # actual route distance
    nodes.loc[o]['geometry'].distance(nodes.loc[d]['geometry']) # straight line distance
    route = nx.shortest_path(G,2653309522,2883191131, weight='length')
    ox.plot_graph_route(G,route,fig_height=10,fig_width=10,)
    '''