#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.optimize import linear_sum_assignment
import numpy as np
from util.utils import get_shortest_path_length
from scipy.spatial.distance import cosine

def km_matching(
        cur_Riders, active_Drivers, all_Riders, G, c_t, route_cache, dis_cache, cost_per_unit,
        cosine_threshold=0, distance_threshold=1.75
):
    if len(cur_Riders) == 0 or len(active_Drivers) == 0: return
    profit_matrix = np.zeros(shape=[len(cur_Riders), len(active_Drivers)]) - 1
    cost_matrix = np.zeros(shape=[len(cur_Riders), len(active_Drivers)])
    mode_matrix = np.zeros(shape=[len(cur_Riders), len(active_Drivers)])
    # total_dis_matrix = np.zeros(shape=[len(cur_Riders), len(active_Drivers)])

    for i in range(len(cur_Riders)):
        rider2 = cur_Riders[i]

        for j in range(len(active_Drivers)):
            driver = active_Drivers[j]
            cur_onboard = driver.riders
            assert len(cur_onboard) < 2, 'Full vehicle'
            assert len(cur_onboard) > 0, 'No onboard rider'
            assert cur_onboard[0] < len(all_Riders), '{},{}'.format(cur_onboard[0], len(all_Riders))
            rider1 = all_Riders[cur_onboard[0]]
            o1, d1 = rider1.o, rider1.d
            o2, d2 = rider2.o, rider2.d
            segments = [(o2, d1), (o2, d2), (d1, d2), (d2, d1), (o1, driver.c_loc), (driver.c_loc, o2)]
            seg_dis_list, exist_list = [], []
            for k in range(len(segments)):
                dis, exist = get_shortest_path_length(G, dis_cache, segments[k][0], segments[k][1])
                # if exist == -1: continue
                seg_dis_list.append(dis)

            '''rules out'''
            # od1 = [rider1.d_lat - rider1.o_lat, rider1.d_lng - rider1.d_lng]
            # od2 = [rider2.d_lat - rider2.o_lat, rider2.d_lng - rider2.d_lng]
            # if 1 - cosine(od1, od2) <= cosine_threshold: continue # TODO: filtering
            dis_list = [seg_dis_list[4], seg_dis_list[5]]
            o1o2 = seg_dis_list[4] + seg_dis_list[5]
            o2d1d2 = seg_dis_list[0] + seg_dis_list[2]
            o2d2d1 = seg_dis_list[1] + seg_dis_list[3]
            if o2d1d2 > o2d2d1: # o1, o2, d2, d1
                total_dis = (o1o2 + o2d2d1)
                # if total_dis / rider1.dis >= distance_threshold: continue
                # if total_dis > rider1.dis + rider2.dis: continue # TODO: filtering
                dis_list += [seg_dis_list[1], seg_dis_list[3]]
            else: # o1, o2, d1, d2
                total_dis = (o1o2 + o2d1d2)
                # if (o1o2 + seg_dis_list[0]) / rider1.dis >= distance_threshold: continue
                # if o2d1d2 / rider2.dis >= distance_threshold: continue # TODO: filtering
                if total_dis > rider1.dis + rider2.dis: continue
                dis_list += [seg_dis_list[0], seg_dis_list[2]]
                mode_matrix[i][j] = 1
            cost = total_dis * cost_per_unit
            profit = rider1.s_price + rider2.s_price - cost
            if profit < rider1.profit_alone + rider2.profit_alone: continue # TODO: filtering

            '''record'''
            cost_matrix[i][j] = cost
            # total_dis_matrix[i][j] = total_dis
            profit_matrix[i][j] = profit

    # print('KM size:{}'.format(profit_matrix.shape))
    row_ind, col_ind = linear_sum_assignment(profit_matrix.clip(min=0), maximize=True)
    # print(len(row_ind), len(col_ind))
    for row, column in zip(row_ind, col_ind):
        value = profit_matrix[row][column]
        if value > 0:
            rider2 = cur_Riders[row]
            driver = active_Drivers[column]
            rider1 = all_Riders[driver.riders[0]]

            driver.append_rider(rider2.r_id, c_t, all_Riders, G, route_cache, dis_cache,
                                mode=int(mode_matrix[row][column]))
            rider2.responded()
            original_total_dis = rider2.dis + rider1.dis
            r2_ratio = rider2.dis / original_total_dis
            r1_ratio = 1 - r2_ratio
            rider2.cost = cost_matrix[row][column] * r2_ratio
            rider1.cost = cost_matrix[row][column] * r1_ratio
            rider2.profit = value * r2_ratio
            rider1.profit = value * r1_ratio
            rider1.shared(rider2.r_id)
            rider2.shared(rider1.r_id)
    return



