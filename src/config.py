import numpy as np

the = {
    'dump': False,
    'go': None,
    'seed': 937162211,
    'bootstrap':512, 
    'conf':0.9, 
    'cliff':.4, 
    'cohen':.35,
    'Fmt': "{:.2f}", 
    'width':40,
    'n_iter': 20,
    'significance_level': 5
}

help = '''USAGE: python main.py  [OPTIONS] [-g ACTION]

OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliffs  cliff's delta threshold      = .147
  -d  --d       different is over sd*d       = .35
  -f  --file    data file                    = ../etc/data/auto2.csv
  -F  --Far     distance to distant          = .95
  -g  --go      start-up action              = nothing
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 10
  -R  --Reuse   child splits reuse a parent pole = true
  -s  --seed    random number seed           = 937162211
    '''

egs = {}

n = 0

top_table = {'all': {'data' : [], 'evals' : 0},
             'sway1': {'data' : [], 'evals' : 0},
             'sway2': {'data' : [], 'evals' : 0},
             'sway3': {'data' : [], 'evals' : 0},
             'sway4': {'data' : [], 'evals' : 0},
             'xpln1': {'data' : [], 'evals' : 0},
             'xpln2': {'data' : [], 'evals' : 0},
             'xpln3': {'data' : [], 'evals' : 0},
             'xpln4': {'data' : [], 'evals' : 0},
             'top': {'data' : [], 'evals' : 0}}

bottom_table = [[['all', 'all'],None],
                [['all', 'sway1'],None],
                [['sway1', 'sway2'],None],
                [['sway1', 'sway3'],None],
                [['sway1', 'sway4'],None],
                [['sway1', 'xpln1'],None],
                [['sway2', 'xpln2'],None],
                [['sway3', 'xpln3'],None],
                [['sway4', 'xpln4'],None],
                [['sway1', 'top'],None]]

hp_grid = {
    'bins': [round(i, 3) for i in list(np.arange(2, 15, 2))],
    'better': ['zitler'],
    'Far': [round(i, 3) for i in list(np.arange(0.5, 1, 0.05))],
    'min': [round(i, 3) for i in list(np.arange(0.5, 1, 0.05))],
    'Max': [round(i, 3) for i in list(np.arange(12, 3000, 500))],
    'p': [round(i, 3) for i in list(np.arange(0.5, 3, 0.25))],
    'rest': [round(i, 3) for i in list(np.arange(1, 10, 1))]
}

hpo_hyperopt_samples = 100
hpo_minimal_sampling_samples = 50