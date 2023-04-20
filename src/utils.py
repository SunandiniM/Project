import sys, re, math, copy, json
from config import *
from pathlib import Path
from sym import SYM
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import itertools, random
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

random.seed(the['seed'])
class Random:
    def __init__(self):
        self.seed = the['seed']
    
    def rint(self, lo=0, hi=1):
        return math.floor(0.5 + rand(lo, hi))

    def set_seed(self, value: int):
        self.seed = value

    def rand(self, lo=0, hi=1):
        self.seed = (16807 * self.seed) % 2147483647
        return lo + (hi - lo) * self.seed / 2147483647

r = Random()
rand = r.rand
rint = r.rint
set_seed = r.set_seed

def per(t, p):
    p = math.floor(((p or 0.5) * len(t)) + 0.5)
    return t[max(0, min(len(t), p) - 1)]

def coerce(s):
    bool_s = s.lower()
    for t in [int, float]:
        try:
            return t(s)
        except ValueError:
            pass
    if bool_s in ["true", "false"]:
        return bool_s == "true"
    return s

def settings(s):
    return dict(re.findall("\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)",s))

def cli(options):
    args = sys.argv[1:]
    for k, v in options.items():
        for n, x in enumerate(args):
            if x == '-'+k[0] or x == '--'+k:
                if v == 'false':
                    v = 'true'
                elif v == 'true':
                    v = 'false'
                else:
                    v = args[n+1]
        options[k] = coerce(v)
    return options

def eg(key, str, fun):
    egs[key] = fun
    global help
    help = help + '  -g '+ key + '\t' + str + '\n'

def rnd(n, nPlaces = 3):
    mult = 10**nPlaces
    return math.floor(n * mult + 0.5) / mult

def csv(sFilename, fun):
    sFilename = Path(sFilename)
    if sFilename.exists() and sFilename.suffix == '.csv':
        t = []
        with open(sFilename.absolute(), 'r', encoding='utf-8') as file:
            for _, line in enumerate(file):
                row = list(map(coerce, line.strip().split(',')))
                t.append(row)
                fun(row)
    else:
        print("File path does not exist OR File not csv, given path: ", sFilename.absolute())
        return

def kap(t, fun):
    u = {}
    what = enumerate(t)
    if type(t) == dict:
        what = t.items()
    for k, v in what:
        v, k = fun(k, v)
        if not k:
            u[len(u)] = v
        else:
            u[k] = v
    return u

def dkap(t, fun):
    u = {}
    for k,v in t.items():
        v, k = fun(k,v) 
        u[k or len(u)] = v
    return u

def cosine(a,b,c):
    if c == 0:
        return 0
    return (a ** 2 + c ** 2 - b ** 2) / (2 * c)

def any(t):
    return t[rint(len(t)) - 1]

def many(t,n):
    u=[]
    for _ in range(1,n+1):
        u.append(any(t))
    return u

def show(node, what, cols, nPlaces, lvl = 0):
  if node:
    print('|..' * lvl, end = '')
    if not node.get('left'):
        print(node['data'].rows[-1].cells[-1])
    else:
        print(int(rnd(100*node['c'], 0)))
    show(node.get('left'), what,cols, nPlaces, lvl+1)
    show(node.get('right'), what,cols,nPlaces, lvl+1)

def deepcopy(t):
    return copy.deepcopy(t)

def oo(t):
    d = t.__dict__
    d['a'] = t.__class__.__name__
    d['id'] = id(t)
    d = dict(sorted(d.items()))
    print(d)

def cliffsDelta(ns1,ns2):
    if len(ns1) > 256:
        ns1 = many(ns1,256)
    if len(ns2) > 256:
        ns2 = many(ns2,256)
    if len(ns1) > 10*len(ns2):
        ns1 = many(ns1,10*len(ns2))
    if len(ns2) > 10*len(ns1):
        ns2 = many(ns2,10*len(ns1))
    n,gt,lt = 0,0,0
    for x in ns1:
        for y in ns2:
            n = n + 1
            if x > y:
                gt = gt + 1
            if x < y:
                lt = lt + 1
    return abs(lt - gt)/n > the['cliffs']

def showTree(node, what, cols, nPlaces, lvl = 0):
  if node:
    print('|.. ' * lvl + '[' + str(len(node['data'].rows)) + ']' + '  ', end = '')
    if not node.get('left') or lvl==0:
        print(node['data'].stats())
    else:
        print('')
    showTree(node.get('left'), what,cols, nPlaces, lvl+1)
    showTree(node.get('right'), what,cols,nPlaces, lvl+1)

def bins(cols,rowss):
    out = []
    for col in cols:
        ranges = {}
        for y,rows in rowss.items():
            for row in rows:
                x = row.cells[col.at]
                if x != '?':
                    k = int(bin(col,x))
                    if not k in ranges:
                        ranges[k] = RANGE(col.at,col.txt,x)
                    extend(ranges[k], x, y)
        ranges = list(dict(sorted(ranges.items())).values())
        r = ranges if isinstance(col, SYM) else mergeAny(ranges)
        out.append(r)
    return out

def bin(col,x):
    if x=='?' or isinstance(col, SYM):
        return x
    tmp = (col.hi - col.lo)/(the['bins'] - 1)
    return  1 if col.hi == col.lo else math.floor(x/tmp + .5)*tmp

def merge(col1,col2):
  new = deepcopy(col1)
  if isinstance(col1, SYM):
      for n in col2.has:
        new.add(n)
  else:
    for n in col2.has:
        new.add(new,n)
    new.lo = min(col1.lo, col2.lo)
    new.hi = max(col1.hi, col2.hi) 
  return new

def RANGE(at,txt,lo,hi=None):
    return {'at':at,'txt':txt,'lo':lo,'hi':lo or hi or lo,'y':SYM()}

def extend(range,n,s):
    range['lo'] = min(n, range['lo'])
    range['hi'] = max(n, range['hi'])
    range['y'].add(s)

def itself(x):
    return x

def value(has,nB = None, nR = None, sGoal = None):
    sGoal,nB,nR = sGoal or True, nB or 1, nR or 1
    b,r = 0,0
    for x,n in has.items():
        if x==sGoal:
            b = b + n
        else:
            r = r + n
    b,r = b/(nB+1/float("inf")), r/(nR+1/float("inf"))
    return b**2/(b+r)


def merge2(col1,col2):
  new = merge(col1,col2)
  if new.div() <= (col1.div()*col1.n + col2.div()*col2.n)/new.n:
    return new

def mergeAny(ranges0):
    def noGaps(t):
        for j in range(1,len(t)):
            t[j]['lo'] = t[j-1]['hi']
        t[0]['lo']  = float("-inf")
        t[len(t)-1]['hi'] =  float("inf")
        return t

    ranges1,j = [],0
    while j <= len(ranges0)-1:
        left = ranges0[j]
        right = None if j == len(ranges0)-1 else ranges0[j+1]
        if right:
            y = merge2(left['y'], right['y'])
            if y:
                j = j+1
                left['hi'], left['y'] = right['hi'], y
        ranges1.append(left)
        j = j+1
    return noGaps(ranges0) if len(ranges0)==len(ranges1) else mergeAny(ranges1)

def firstN(sortedRanges,scoreFun):
    print("")
    def function(r):
        print(r['range']['txt'],r['range']['lo'],r['range']['hi'],rnd(r['val']),r['range']['y'].has)
    _ = list(map(function, sortedRanges))
    print()
    first = sortedRanges[0]['val']
    def useful(range):
        if range['val']>.05 and range['val']> first/10:
            return range
    sortedRanges = [x for x in sortedRanges if useful(x)]
    most,out = -1, -1
    for n in range(1,len(sortedRanges)+1):
        slice = sortedRanges[0:n]
        slice_range = [x['range'] for x in slice]
        tmp,rule = scoreFun(slice_range)
        if tmp and tmp > most:
            out,most = rule,tmp
    return out,most

def prune(rule, maxSize):
    n=0
    for txt,ranges in rule.items():
        n = n+1
        if len(ranges) == maxSize[txt]:
            n=n+1
            rule[txt] = None
    if n > 0:
        return rule
    
def get_avgs_from_data_list(data_obj_list):
    avgs = {}
    # For each data_obj, get sum of mid/mode
    for data_obj in data_obj_list:
        for k,v in data_obj.stats().items():
            avgs[k] = avgs.get(k,0) + v
    # Convert sums to averages
    for k,v in avgs.items():
        avgs[k] = round(avgs[k]/the['n_iter'], 2)
    return avgs

def preprocess_data(file, DATA):
    df = pd.read_csv(file)
    df = impute_missing_values(df)
    label_encoding(df)
    file = file.replace('.csv', '_preprocessed.csv')
    df.to_csv(file, index=False)
    return DATA(file)

def impute_missing_values(df):
    for i in df.columns[df.isnull().any(axis=0)]:
        df[i].fillna(df[i].mean(),inplace=True)
    for col in df.columns[df.eq('?').any()]:
        df[col] =df[col].replace('?', np.nan)
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].mean())
    return df

def label_encoding(df):
    syms = [col for col in df.columns if col.strip()[0].islower() and df[col].dtypes == 'O']
    le = LabelEncoder()
    for sym in syms:
        col = le.fit_transform(df[sym])
        df[sym] = col.copy()

def better_dicts(dict1, dict2, data):
    s1, s2, ys, x, y = 0, 0, data.cols.y, None, None

    for col in ys:
        x = dict1.get(col.txt)
        y = dict2.get(col.txt)
        x = col.norm(x)
        y = col.norm(y)
        s1 = s1 - math.exp(col.w * (x - y) / len(ys))
        s2 = s2 - math.exp(col.w * (y - x) / len(ys))

    return s1 / len(ys) < s2 / len(ys)

def hpo_minimal_sampling_params(DATA, XPLN, selects, showRule):
    most_optimial_sway_hps = []
    least_sway_evals = -1
    least_xpln_evals = -1
    best_sway = ""
    best_xpln = ""
    
    combs = list(itertools.product(*hp_grid.values()))
    all_hp_combs = []
    for params in combs:
        all_hp_combs.append(dict(zip(hp_grid.keys(), params)))
    hp_combs_sample = random.sample(all_hp_combs, the['hpo_minimal_sampling_samples'])
                           
    for hps in hp_combs_sample:
        the.update(hps)
        data=DATA(the["file"]) 
        best,rest, evals = data.sway() 
        xp = XPLN(best, rest)
        rule,_= xp.xpln(data,best,rest)
        data1= DATA(data,selects(rule,data.rows))
        current_sway =  best.stats(best.cols.y, 2, 'mid')
        top,_ = data.betters(len(best.rows))
        top = data.clone(top)

        if(best_xpln == ""):
            best_xpln = data1.stats(data1.cols.y, 2, 'mid')
            best_sway = best.stats(best.cols.y, 2, 'mid')
            most_optimial_sway_hps = hps
            least_sway_evals = evals
            least_xpln_evals = evals
        else:
            if better_dicts(current_sway, best_sway, data):
                best_sway = current_sway
                most_optimial_sway_hps = hps
                least_sway_evals = evals
    the.update(most_optimial_sway_hps)
    print('--------------- HPO Minimal Sampling Results ---------------')
    print('Best Params: ', most_optimial_sway_hps)
    print("\n-----------\nexplain=", showRule(rule))
    print("all               ",data.stats(data.cols.y, 2, 'mid'))
    print("sway with",least_sway_evals,"evals",best_sway)
    print("xpln on",least_xpln_evals,"evals",best_xpln)
    print("sort with",len(data.rows),"evals",top.stats(top.cols.y, 2, 'mid'))

def hpo_hyperopt_params(DATA, XPLN, selects, showRule):
    def hyperopt_objective(params):
        current_sway = {}
        current_xpln = {}
        top = {}
        sum = 0
        the.update(params)
        data=DATA(the["file"]) 
        best,rest, evals = data.sway() 
        xp = XPLN(best, rest)
        rule,_= xp.xpln(data,best,rest)
        if rule != -1:
            data1= DATA(data,selects(rule,data.rows))
            current_sway =  best.stats(best.cols.y, 2, 'mid')
            current_xpln = data1.stats(data1.cols.y, 2, 'mid')
            top,_ = data.betters(len(best.rows))
            top = data.clone(top)
            ys = data.cols.y
            for col in ys:
                x = current_sway.get(col.txt)
                sum += x * col.w
        return {"loss": sum, "status": STATUS_OK, "data": data, "evals": evals, "rule": rule, "sway": current_sway, "xpln": current_xpln, "top": top, "params": params}
    
    space = {}
    trials = Trials()
    for key in hp_grid.keys():
        space[key] = hp.choice(key, hp_grid[key])
    
    best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=the['hpo_hyperopt_samples'], trials=trials)
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_loss)
    best_trial = trials.trials[best_ind]
    print('--------------- HPO Hyperopt Results ---------------')
    print('Best Params: ', best_trial['result']['params'])
    print("\n-----------\nexplain=", showRule(best_trial['result']['rule']))
    print("all               ",best_trial['result']['data'].stats(best_trial['result']['data'].cols.y, 2, 'mid'))
    print("sway with",best_trial['result']['evals'],"evals",best_trial['result']['sway'])
    print("xpln on",best_trial['result']['evals'],"evals",best_trial['result']['xpln'])
    print("sort with",len(best_trial['result']['data'].rows),"evals",best_trial['result']['top'].stats(best_trial['result']['top'].cols.y, 2, 'mid'))
    print()

def run_stats(data, top_table):
    print('MWU and KW significance level: ', the['significance_level']/100)
    all =  top_table['all']
    sway1 = top_table['sway1']
    sway2 = top_table['sway2 (pca)']
    sway3 = top_table['sway3 (agglo)']
    sway4 = top_table['sway4 (kmeans)']
    sways = ['sway1', 'sway2', 'sway3', 'sway4']
    mwu_sways = []
    kw_sways = []

    xpln1 = top_table['xpln1']
    xpln2 = top_table['xpln2 (pca+kdtree)']
    xpln3 = top_table['xpln3 (agglo+kdtree)']
    xpln4 = top_table['xpln4 (kmeans+kdtree)']
    xplns = ['xpln1', 'xpln2', 'xpln3', 'xpln4']
    mwu_xplns = []
    kw_xplns = []

    kw_sway_p_values = []
    kw_xpln_p_values = []

    taxes = {'samp tax1': [], 'xpln tax1': [], 'samp tax2': [], 'xpln tax2': [], 'samp tax3': [], 'xpln tax3': [], 'samp tax4': [], 'xpln tax4': []}

    for col in data.cols.y:
        sway_avgs = [sway1['avg'][col.txt], sway2['avg'][col.txt], sway3['avg'][col.txt], sway4['avg'][col.txt]]
        xpln_avgs = [xpln1['avg'][col.txt], xpln2['avg'][col.txt], xpln3['avg'][col.txt], xpln4['avg'][col.txt]]

        for idx in range(len(xpln_avgs)):
            taxes['samp tax' + str(idx + 1)].append(round(all['avg'][col.txt] - sway_avgs[idx], 2))
            taxes['xpln tax'  + str(idx + 1)].append(round(sway_avgs[idx] - xpln_avgs[idx], 2))

            
        if col.w == -1:
            best_avg = min(sway_avgs)
        else:
            best_avg = max(sway_avgs)
        best_sway = sways[sway_avgs.index(best_avg)]

        for best in sway1['data']:
            sway1_col = [row.cells[col.at] for row in best.rows]
        for best in sway2['data']:
            sway2_col = [row.cells[col.at] for row in best.rows]
        for best in sway3['data']:
            sway3_col = [row.cells[col.at] for row in best.rows]
        for best in sway4['data']:
            sway4_col = [row.cells[col.at] for row in best.rows]

        _, p_value_sways = kruskal(sway1_col, sway2_col, sway3_col, sway4_col)
        kw_sway_p_values.append(p_value_sways/100)
        # if p_value < the['kw_significance']:
        #     top_table['kw_significant'].append('yes')
        # else:
        #     top_table['kw_significant'].append('no')
        
        groups = [sway1_col, sway2_col, sway3_col, sway4_col]
        num_groups = len(groups)
        p_values_mwu = np.zeros((num_groups, num_groups))
        p_values_kruskal = np.zeros((num_groups, num_groups))

        for i in range(num_groups):
            for j in range(i+1, num_groups):
                _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_values_mwu[i, j] = p
                p_values_mwu[j, i] = p
                _, p_value_kruskal = kruskal(sway1_col, sway2_col, sway3_col, sway4_col)
                p_values_kruskal[i, j] = p_value_kruskal
                p_values_kruskal[j, i] = p_value_kruskal

        # print('Pairwise Mann-Whitney U tests')
        # print(pd.DataFrame(p_values_mwu, index=sways, columns=sways))
        # print()

        # Apply Bonferroni correction for multiple comparisons
        adjusted_p_values = multipletests(p_values_mwu.ravel(), method='fdr_bh')[1].reshape(p_values_mwu.shape)
        post_hoc = pd.DataFrame(adjusted_p_values, index=sways, columns=sways)

        # print("Pairwise Mann-Whitney U tests with Benjamini/Hochberg correction:")
        # print(post_hoc)
        # print()

        krusal_df = pd.DataFrame(p_values_kruskal, index=sways, columns=sways)
        # print("Pairwise Kruskal Wallis:")
        # print(krusal_df)
        # print()

        mwu_sig = set(post_hoc.iloc[list(np.where(post_hoc >= the['significance_level'])[0])].index)
        if len(mwu_sig) == 0:
            mwu_sways.append([best_sway])
        else:
            mwu_sways.append(list(mwu_sig))

        kw_sig = set(krusal_df.iloc[list(np.where(krusal_df >= the['significance_level'])[0])].index)
        if len(kw_sig) == 0:
            kw_sways.append([best_sway])
        else:
            kw_sways.append(list(kw_sig))

    for col in data.cols.y:
        xpln_avgs = [xpln1['avg'][col.txt], xpln2['avg'][col.txt], xpln3['avg'][col.txt], xpln4['avg'][col.txt]]
        if col.w == -1:
            best_avg = min(xpln_avgs)
        else:
            best_avg = max(xpln_avgs)
        best_xpln = xplns[xpln_avgs.index(best_avg)]

        for best in xpln1['data']:
            xpln1_col = [row.cells[col.at] for row in best.rows]
        for best in xpln2['data']:
            xpln2_col = [row.cells[col.at] for row in best.rows]
        for best in xpln3['data']:
            xpln3_col = [row.cells[col.at] for row in best.rows]
        for best in xpln4['data']:
            xpln4_col = [row.cells[col.at] for row in best.rows]
        
        _, p_value_xplns = kruskal(xpln1_col, xpln2_col, xpln3_col, xpln4_col)
        kw_xpln_p_values.append(p_value_xplns/100)
        groups = [xpln1_col, xpln2_col, xpln3_col, xpln4_col]
        num_groups = len(groups)
        p_values_mwu = np.zeros((num_groups, num_groups))
        p_values_kruskal = np.zeros((num_groups, num_groups))

        for i in range(num_groups):
            for j in range(i+1, num_groups):
                _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_values_mwu[i, j] = p
                p_values_mwu[j, i] = p
                _, p_value_kruskal = kruskal(xpln1_col, xpln2_col, xpln3_col, xpln4_col)
                p_values_kruskal[i, j] = p_value_kruskal
                p_values_kruskal[j, i] = p_value_kruskal

        adjusted_p_values = multipletests(p_values_mwu.ravel(), method='fdr_bh')[1].reshape(p_values_mwu.shape)
        post_hoc = pd.DataFrame(adjusted_p_values, index=xplns, columns=xplns)
        krusal_df = pd.DataFrame(p_values_kruskal, index=xplns, columns=xplns)

        mwu_sig = set(post_hoc.iloc[list(np.where(post_hoc >= the['significance_level'])[0])].index)
        if len(mwu_sig) == 0:
            mwu_xplns.append([best_xpln])
        else:
            mwu_xplns.append(list(mwu_sig))

        kw_sig = set(krusal_df.iloc[list(np.where(krusal_df >= the['significance_level'])[0])].index)
        if len(kw_sig) == 0:
            kw_xplns.append([best_xpln])
        else:
            kw_xplns.append(list(kw_sig))

    return mwu_sways, kw_sways, mwu_xplns, kw_xplns, taxes, kw_sway_p_values, kw_xpln_p_values
