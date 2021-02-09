#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setproctitle import setproctitle as ptitle
from util.arg_parser import init_parser
from carpool.carpool_simulation import carpool_running
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    print('################ START TIME: {} ################'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))

    parser = init_parser()
    args = parser.parse_args()
    temp = {'OnFair': 1, 'OnFair-Exploit': 2, 'Fixed': 3, 'ProfitOnly': 4}
    if args.scheme not in temp.keys(): raise RuntimeError('unknown scheme')
    ptitle('{}K{}o{}w{}{}'.format(temp[args.scheme], args.K, args.omega, args.week, args.isLinear))

    args_ = {
        'omega': args.omega,
        'K': args.K,
        'constant_discount':args.a,
        'running_times': args.running_times,
        'scheme': args.scheme,
        'isLinear':args.isLinear,
        'week':args.week,
        'penalty':args.penalty,
        'ratio':args.ratio,
        'preset_cost':args.preset_cost,
        'user_based':args.user_based
    }

    carpool_running(**args_)

    print('################ END TIME: {} ################\n'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))