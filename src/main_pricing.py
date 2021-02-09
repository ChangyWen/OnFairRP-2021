#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setproctitle import setproctitle as ptitle
from util.arg_parser import init_parser
from optimization.optimize_solve import fair, fixed, profitOnly
from datetime import datetime
import time

if __name__ == '__main__':
    print('################ START TIME: {} ################'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))

    parser = init_parser()
    args = parser.parse_args()
    temp = {'OnFair':1, 'OnFair-Exploit':2, 'Fixed':3, 'ProfitOnly':4}
    if args.scheme not in temp.keys(): raise RuntimeError('unknown scheme')
    ptitle('{}K{}o{}w{}{}'.format(temp[args.scheme], args.K, args.omega, args.week, args.isLinear))
    args_ = {
        'omega_': args.omega,
        'isLinear': args.isLinear,
        'discount_a':args.a,
        'week':args.week,
        'distribution_file': '../data/_wkday_in_wk{}_sum_h3.csv'.format(args.week)
    }
    s = time.time()
    if args.scheme == 'Fixed':
        fixed(**args_)
    elif args.scheme == 'OnFair':
        args_['K_similar_const'] = args.K
        fair(**args_)
    elif args.scheme == 'ProfitOnly':
        args_['penalty_'] = args.penalty
        args_['ratio_'] = args.ratio
        profitOnly(**args_)
    print('Consumed Time:{}'.format(time.time() - s))

    print('################ END TIME: {} ################'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))