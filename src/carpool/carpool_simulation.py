#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import osmnx as ox
import numpy as np
from driver_rider import Driver, Rider
from km_match import km_matching
from collections import defaultdict
from random import sample
from util.utils import demand_func, init_data, load_obj, complement_sum, pre_data
from datetime import datetime
import shutil
import os
from util.get_unfairness_info import get_unfair_list

def carpool_running(
        omega, K, constant_discount, running_times, scheme, isLinear, week, penalty, ratio, preset_cost, user_based,
        interval=5, fee_per_unit=0.003, cost_per_unit = 0.003 * 0.25, speed = 6.94,
):
    # TODO: preset cost and dynamic cost
    ox.config(log_console=False, use_cache=True)
    G = ox.graph_from_bbox(
        north=30.730364 + 0.02, south=30.650341 - 0.02,
        west=104.038645 - 0.02, east=104.13389 + 0.02,
        network_type='drive'
    )
    demand_type = 'linear' if isLinear else 'exp'
    if scheme in ['OnFair', 'OnFair-Exploit']:
        discount_col = 'OnFair_{}_K{}'.format(demand_type, K)
        output_file = 'OnFair_{}_K{}_o{}_a{}_wk1.txt'.format(demand_type, K, omega, constant_discount)
    elif scheme == 'Fixed':
        discount_col = 'Fixed_{}'.format(demand_type)
        output_file = 'Fixed_{}_o{}_a{}_wk1.txt'.format(demand_type, omega, constant_discount)
    else:
        discount_col = 'ProfitOnly_{}_p{}_r{}'.format(demand_type, penalty, ratio)
        output_file = 'ProfitOnly_{}_o{}_a{}_wk1_p{}_r{}.txt'.format(demand_type, omega, constant_discount, penalty, ratio)

    sum_df = complement_sum(output_file, discount_col)
    all_riders = pre_data(sum_df)
    nums_riders = len(all_riders)
    if scheme == 'ProfitOnly' and K < 8.0 and penalty > 0:
        unfair_ratio_list = []
        od_weight = 16.9071
        if user_based:
            user_groups = all_riders.groupby(['user_ids'])
        else:
            uf_directory = '../data/uf_{}_K{}_wk2'.format(discount_col, K)
            get_unfair_list(K, discount_col, uf_directory, all_riders)

        for i in range(nums_riders):
            unfair_ratio = 0
            discount = all_riders.loc[i, discount_col]
            ods = all_riders.loc[i, ['olat', 'olng', 'dlat', 'dlng']].to_numpy().astype(np.float64)
            tau = all_riders.loc[i, 'tau']
            if user_based:
                user_orders = user_groups.get_group(all_riders.loc[i, 'user_ids'])
                pre_orders = user_orders[user_orders.index < i]
                if len(pre_orders.index) > 0:
                    pre_discounts = pre_orders[discount_col].values
                    pre_ods = pre_orders[['olat', 'olng', 'dlat', 'dlng']].to_numpy().astype(np.float64)
                    pre_taus = pre_orders['tau'].values
                    od_dis = np.sum(np.square(ods)) + np.sum(np.square(pre_ods), axis=1) - 2 * np.sum(np.multiply(ods, pre_ods), axis=1)
                    tau_dis = np.not_equal(tau, pre_taus).astype(int)
                    sim = od_weight * od_dis + tau_dis
                    unfair_ratio = np.sum(K * sim + pre_discounts < discount) / len(pre_orders.index)
            else:
                if i % 1500 == 0: unfair_info = load_obj(uf_directory + '/{}.pkl'.format(i // 1500))
                len_to_be_compared = i
                if len_to_be_compared > 0: unfair_ratio = len(unfair_info[i % 1500]) / len_to_be_compared
            unfair_ratio_list.append(unfair_ratio)

        if user_based: del user_groups
        else: del unfair_info

    route_cache, dis_cache = init_data(
        route_cache_loc='../data/route_cache.pkl',
        dis_cache_loc='../data/dis_cache.pkl'
    )
    total_profits, total_costs, share_rates, nums_S_riders, cumulative_profits = [], [], [], [], []  # running_times --> list
    t_range = range(0, int(432000 / interval)) # 432000 per weeks TODO
    mid_t = int(len(t_range) / 2)

    if isLinear:
        seeds = [13941, 70284, 59002, 76847, 55872, 32590, 54643, 20434, 91961, 45339]
        learning_transition_phases = [17248, 17749, 17673, 17188, 17246, 17256, 17721, 17724, 17214, 17419]
    else:
        # seeds = [97362, 78739, 86541, 33168, 6498, 69609, 32871, 1111, 15280, 61313]
        # learning_transition_phases = [45500, 45446, 45251, 45037, 45086, 45195, 45469, 45363, 45022, 45500]
        seeds = [59559, 78656, 25895, 64329, 78361, 23774, 7457, 39713, 2113, 11001]
        learning_transition_phases = [13206, 13468, 13956, 13610, 13323, 13668, 13229, 13927, 13244, 13396]

    for _ in range(running_times):
        seed = seeds[_]
        np.random.seed(seed)
        rands = np.random.random(nums_riders)
        if scheme == 'OnFair': pre_phase = learning_transition_phases[_]
        # if scheme == 'ProfitOnly': unfair_ratio_list_temp = []
        '''initialize all_Riders'''
        all_Riders = []
        time_step_dict = defaultdict(set)
        r_id = 0
        SRide_list = []
        print('### {} {} K{} p{} r{} {} Simulation:{} Start ###'.format(scheme, demand_type, K, penalty, ratio, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), _))
        for i in range(nums_riders):
            gamma = all_riders.loc[i, 'gamma']
            discount = all_riders.loc[i, discount_col]
            if scheme == 'OnFair' and i < pre_phase: discount = constant_discount

            prob_SRide = demand_func(gamma=gamma, discount=discount, omega=omega, isLinear=isLinear)
            unfair_penalty = 0
            if scheme == 'ProfitOnly' and K < 8.0 and penalty > 0:
                unfair_penalty = unfair_ratio_list[i] * penalty
            isSRide = rands[i] <= prob_SRide - unfair_penalty
            SRide_list.append(isSRide)
            if isSRide:
                o_lat = all_riders.loc[i, 'olat']
                o_lng = all_riders.loc[i, 'olng']
                d_lat = all_riders.loc[i, 'dlat']
                d_lng = all_riders.loc[i, 'dlng']
                o = all_riders.loc[i, 'o']
                d = all_riders.loc[i, 'd']
                time_step = all_riders.loc[i, 'timestamp']
                dis = 1 if o == d else dis_cache[(o, d)]
                rider = Rider(
                    r_id, time_step, discount, o, d, dis, o_lat, o_lng, d_lat, d_lng, fee_per_unit, cost_per_unit
                )
                time_step_dict[rider.time_step].add(r_id)
                all_Riders.append(rider)
                r_id += 1

        print('### {}  all Riders initiated ({}) ###'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), len(all_Riders)))

        active_Drivers = []
        for t in t_range:
            if t == mid_t: print('### {}  {}-round ({} rounds in total) ###'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), t, len(t_range)))
            cur_Riders = []
            cur_time = (t + 1) * interval
            for t_ in range(t * interval, (t + 1) * interval):
                for r_id in time_step_dict[t_]:
                    cur_Riders.append(all_Riders[r_id])

            if (not active_Drivers) or preset_cost:
            #     for driver in active_Drivers:
            #         if driver.isFull(): raise RuntimeError('full vehicle!!!')
                for rider in cur_Riders:
                    r_id = rider.r_id
                    o = rider.o
                    d = rider.d
                    active_Drivers.append(Driver(r_id, cur_time, o, route_cache[(o, d)].copy(), speed))
                    rider.responded()
            else:
                for driver in active_Drivers:
                    driver.update(cur_time, all_Riders, G, route_cache, dis_cache)
                to_remove = []
                for driver in active_Drivers:
                    if driver.isFull():
                        to_remove.append(driver)
                    elif driver.isEmpty():
                        to_remove.append(driver)
                for driver in to_remove:
                    active_Drivers.remove(driver)

                km_matching(cur_Riders, active_Drivers, all_Riders, G, cur_time, route_cache, dis_cache, cost_per_unit)

                left_riders = []
                for rider in cur_Riders:
                    if not rider.isResponded():
                        left_riders.append(rider)
                for rider in left_riders:
                    r_id = rider.r_id
                    o = all_Riders[r_id].o
                    d = all_Riders[r_id].d
                    active_Drivers.append(Driver(r_id, cur_time, o, route_cache[(o, d)].copy(), speed))
                    rider.responded()

        nums_shared, total_profit, total_cost = 0, 0, 0
        round_profit_list = []
        j = 0
        for i in range(nums_riders):
            isS = SRide_list[i]
            if isS:
                total_profit += all_Riders[j].profit
                total_cost += all_Riders[j].cost
                nums_shared = nums_shared + 1 if all_Riders[j].isShared() else nums_shared
                # if not all_Riders[j].isResponded(): raise RuntimeError('not responded')
                # if all_Riders[j].isShared() and all_Riders[j].shared_id > j:
                #     shared_rider_id = all_Riders[j].shared_id
                #     assert j == all_Riders[shared_rider_id].shared_id and all_Riders[shared_rider_id].isShared(), \
                #         '{} {} {} {}'.format(j, all_Riders[j].r_id, all_Riders[shared_rider_id].shared_id, all_Riders[shared_rider_id].isShared())
                #     s_profit = all_Riders[j].profit + all_Riders[shared_rider_id].profit
                #     assert round(s_profit,3) >= round(all_Riders[j].profit_alone + all_Riders[shared_rider_id].profit_alone, 3), '{} {} {}'.format(s_profit, all_Riders[j].profit_alone, all_Riders[shared_rider_id].profit_alone)
                j += 1
            round_profit_list.append(total_profit)
        assert j == len(all_Riders), '{}_{}'.format(j, len(all_Riders))
        share_rate = nums_shared / len(all_Riders)

        total_profits.append(total_profit)
        total_costs.append(total_cost)
        share_rates.append(share_rate)
        nums_S_riders.append(len(all_Riders))
        cumulative_profits.append(round_profit_list)

        print('### {} {} K{} p{} r{} {} Simulation:{} End ###'.format(scheme, demand_type, K, penalty, ratio, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), _))
        # print('### {} {}_{}_K{} Simulation:{} End ###'.format(scheme, demand_type, K, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), _))

    '''save results'''
    data1 = [
        [nums_riders] * running_times,
        nums_S_riders, share_rates, total_profits, total_costs
    ]

    if scheme == 'ProfitOnly':
        if K < 8.0 and penalty > 0:
            data1.append([np.mean(unfair_ratio_list)] * running_times)
            if not user_based: shutil.rmtree(uf_directory)
            print('unfair_ratios (mean):', data1[-1])
        else:
            data1.append([0 for i in range(running_times)])
        discount_col += '_K{}'.format(K)
    if scheme == 'OnFair-Exploit': discount_col = 'OnFair-Exploit' + discount_col[6:]
    directory = '../output/simulation_res/{}'.format(discount_col)
    if not preset_cost: directory += '-dynamic'
    if user_based: directory += '-user_based'
    os.makedirs(directory, exist_ok=True)

    print('nums_riders:', nums_riders)
    print('nums_S_riders:', nums_S_riders)
    print('share_rates:', share_rates)
    print('total_profits:', total_profits)
    print('total_costs:', total_costs)

    np.savetxt(fname=directory + '/statistics.txt', X=np.array(data1).reshape(-1, running_times).transpose())
    np.savetxt(fname=directory + '/cumulative_profits.txt', X=np.array(cumulative_profits).transpose())
    if scheme == 'ProfitOnly' and K < 8.0 and penalty > 0:
        np.savetxt(fname=directory + '/unfair_ratios.txt', X=np.array(unfair_ratio_list).transpose())
