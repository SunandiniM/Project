#!/usr/bin/env python3

__author__ = "NCSU CSC 591 021 Spring 23 Group-3"
__version__ = "1.0.0"
__license__ = "MIT"

from utils import *
from test_hw6 import *
from xpln import *
from data import DATA
from hw7 import *
from tabulate import tabulate

def main():
    y,n,saved = 0,0,deepcopy(the)
    for k,v in cli(settings(help)).items():
        the[k] = v
        saved[k] = v
    if the['help'] == True:
        print(help)
    else:
        count = 0
        while count < the['n_iter']:
            data = DATA(the['file'])
            data2 = impute_missing_values(the['file'], DATA)
            best,rest,evals = data.sway()
            xp = XPLN(best, rest)
            rule,_= xp.xpln(data,best,rest)
            if rule != -1:
                betters, _ = data.betters(len(best.rows))
                top_table['top']['data'].append(DATA(data,betters))
                top_table['xpln1']['data'].append(DATA(data,selects(rule,data.rows)))
                top_table['xpln2']['data'].append(DATA(data,selects(rule,data.rows)))
                top_table['all']['data'].append(data)
                top_table['sway1']['data'].append(best)
                top_table['sway2']['data'].append(best)
                top_table['all']['evals'] += 0
                top_table['sway1']['evals'] += evals
                top_table['sway2']['evals'] += evals
                top_table['xpln1']['evals'] += evals
                top_table['xpln2']['evals'] += evals
                top_table['top']['evals'] += len(data.rows)
                
                for i in range(len(bottom_table)):
                    [base, diff], result = bottom_table[i]
                    if result == None:
                        bottom_table[i][1] = ['=' for _ in range(len(data.cols.y))]
                    for k in range(len(data.cols.y)):
                        if bottom_table[i][1][k] == '=':
                            y0, z0 = top_table[base]['data'][count].cols.y[k],top_table[diff]['data'][count].cols.y[k]
                            is_equal = bootstrap(y0.vals(), z0.vals()) and cliffsDelta(y0.vals(), z0.vals())
                            if not is_equal:
                                bottom_table[i][1][k] = '≠'
                count += 1
        
        with open(the['file'].replace('/data', '/out').replace('.csv', '.out'), 'w') as outfile:
            headers = [y.txt for y in data.cols.y]
            table = []

            for k,v in top_table.items():
                stats = [k] + [stats_average(v['data'])[y] for y in headers]
                stats += [v['evals']/the['n_iter']]
                table.append(stats)
            
            print(tabulate(table, headers=headers+["n_evals avg"],numalign="right"))
            print()
            outfile.write(tabulate(table, headers=headers+["n_evals avg"],numalign="right"))
            outfile.write('\n')

            table=[]
            for [base, diff], result in bottom_table:
                table.append([f"{base} to {diff}"] + result)
            print(tabulate(table, headers=headers,numalign="right"))
            outfile.write(tabulate(table, headers=headers,numalign="right"))

        for what, fun in egs.items():
            if the['go'] == 'all' or the['go'] == what:
                for k,v in saved.items():
                    the[k] = v
                Seed = the['seed']
                print('▶️ ',what,('-')*(60))
                if egs[what]() == False:
                    n += 1
                    print('❌ fail:', what)
                else:
                    y += 1
                    print('✅ pass:', what)
    if y+n>0:
        print("🔆",{'pass' : y, 'fail' : n, 'success' :100*y/(y+n)//1})
    sys.exit(n)

if __name__ == '__main__':
    eg('the', 'show options', test_the)
    eg('some', 'demo of reservoir sampling', test_some)
    eg('nums', 'demo of NUM', test_num)
    eg('sym', 'demo SYMS', test_sym)
    eg('csv', 'reading csv files', test_csv)
    eg('data', 'showing DATA sets', test_data)
    eg('clone', 'replicate structure of a DATA', test_clone)
    eg('cliffs', 'start tests', test_cliffs)
    eg('dist', 'distance test', test_dist)
    eg('tree', 'make snd show tree of clusters', test_tree)
    eg('sway', 'optimizing', test_sway)
    eg('bins', 'find deltas between best and rest', test_bins)
    eg('xpln', 'explore explanation sets', test_xpln)
    main()