the = {
    'dump': False,
    'go': None,
    'seed': 937162211,
    'bootstrap':512, 
    'conf':0.05, 
    'cliff':.4, 
    'cohen':.35,
    'Fmt': "{:.2f}", 
    'width':40,
    'n_iter': 20
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
             'xpln1': {'data' : [], 'evals' : 0}, 
             'top': {'data' : [], 'evals' : 0}}

bottom_table = [[['all', 'all'],None], 
                [['all', 'sway1'],None],  
                [['sway1', 'xpln1'],None],  
                [['sway1', 'top'],None]]