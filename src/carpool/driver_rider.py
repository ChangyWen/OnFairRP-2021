#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util.utils import get_shortest_path, get_shortest_path_length

class Driver(object):
    def __init__(self, r_id, c_t, c_loc, route, speed):
        self.riders = [r_id]
        self.c_t = c_t
        self.c_loc = c_loc
        if len(route) == 0:
            raise RuntimeError('self.route = []')
        self.route = route
        self.speed = speed

    def isFull(self):
        return True if len(self.riders) >= 2 else False

    def isEmpty(self):
        return True if len(self.riders) == 0 else False

    def append_rider(self, r_id, t, all_Riders, G, route_cache, dis_cache, mode):
        self.riders.append(r_id)
        self.update(t, all_Riders, G, route_cache, dis_cache, mode=mode, newRider=True)
        # self.update_route(t, all_Riders, mode=mode, G=G, newRider=True, route_cache=route_cache, dis_cache=dis_cache)

    def update(self, t, all_Riders, G, route_cache, dis_cache, mode=None, newRider=False):
        if self.isFull(): return
        self.update_route_and_riders(t, all_Riders, G, route_cache, dis_cache, mode=mode, newRider=newRider)

    def update_route_and_riders(self, t, all_Riders, G, route_cache, dis_cache, mode=None, newRider=False):
        if newRider:
            rider2 = all_Riders[self.riders[1]]
            rider1 = all_Riders[self.riders[0]]
            d1 = rider1.d
            o2 = rider2.o
            d2 = rider2.d

            route1, _ = get_shortest_path(G, route_cache, self.c_loc, o2)
            if mode == 0:
                route2, _ = get_shortest_path(G, route_cache, o2, d2)
                route3, _ = get_shortest_path(G, route_cache, d2, d1)
            else:
                route2, _ = get_shortest_path(G, route_cache, o2, d1)
                route3, _ = get_shortest_path(G, route_cache, d1, d2)

            self.route = route1[:-1] + route2[:-1] + route3

        # if len(self.route) == 0: return
        while self.route:
            time = get_shortest_path_length(G, dis_cache, self.c_loc, self.route[0])[0]/ self.speed
            if t >= self.c_t + time:
                self.c_loc = self.route.pop(0)
                self.c_t += time
                self.update_riders(all_Riders)
            else:
                # self.c_t = t # note that self.c_t is the time that vehicle arrives at self.route[0]
                break

    def update_riders(self, all_Riders):
        to_remove = []
        for r_id in self.riders:
            if self.c_loc == all_Riders[r_id].d:
                to_remove.append(r_id)
        for r_id in to_remove:
            self.riders.remove(r_id)


class Rider(object):
    def __init__(
            self, r_id, time_step, discount, o, d, distance, o_lat, o_lng, d_lat, d_lng, fee_per_unit, cost_per_unit
    ):
        self.r_id = r_id
        self.time_step = time_step
        self.respond_status = False
        self.discount = discount
        self.dis = distance
        # self.gamma = gamma
        self.o = o
        self.d = d
        self.o_lat = o_lat
        self.o_lng = o_lng
        self.d_lat = d_lat
        self.d_lng = d_lng
        # self.tau = tau
        self.shared_id = None
        self.shared_status = False
        self.p_price = self.dis * fee_per_unit
        self.cost = self.dis * cost_per_unit
        self.s_price = self.p_price * self.discount
        self.profit = self.s_price - self.cost
        self.profit_alone = self.profit
        # self.isSRide = self.isSRide_(omega, isLinear)

    def isResponded(self):
        return self.respond_status

    def responded(self):
        self.respond_status = True

    def isShared(self):
        return self.shared_status

    def shared(self, shared_id):
        self.shared_id = shared_id
        self.shared_status = True

    # def isSRide_(self, omega, isLinear):
    #     prob_SRide = demand_func(discount=self.discount, omega=omega, tau=self.tau, isLinear=isLinear)
    #     return uniform(0, 1) <= prob_SRide
