#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def init_parser():
    parser = argparse.ArgumentParser(description='OnlineFairRSPricing')

    parser.add_argument('--OMEGA', nargs='*', help='set of values of omegas, (defalut=[0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58])', required=False)
    parser.add_argument('--eta_peak', type=float, default=0.8, help='eta of peak hour (default=0.8)')
    parser.add_argument('--eta_off', type=float, default=0.6, help='eta of off hour (default=0.6)')
    parser.add_argument('--omega', type=float, default=0.4, help='omega (default=0.4)')

    parser.add_argument('--running_times', type=int, default=1, help='running times of data every day (default=1)')
    parser.add_argument('--K', type=float, default=0.32, help='constant K (default=0.32)')
    parser.add_argument('--sigma', type=float, default=0.08, help='sigma (default=0.08)')
    parser.add_argument('--scheme', type=str, default='OnFair', help='scheme name (default=\'OnFair\'), [OnFair, ProfitOnly, Fixed, OnFair-Exploit]')

    parser.add_argument('--a', type=float, default=0.99, help='constant discount or default discount of the Optimization (default=0.99)')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold of learning (default=1.0)')
    parser.add_argument('--isLinear', action='store_true', default=False)
    parser.add_argument('--preset_cost', action='store_true', default=False)
    parser.add_argument('--user_based', action='store_true', default=False)
    parser.add_argument('--week', type=int, default=1, help='which week\'s data to use (default=1)')

    parser.add_argument('--penalty', type=float, default=0.05, help='the penalty of unfairness in demand function (default=0.05)')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio in the objective function of ProfitOnly (default=1.0)')
    return parser
