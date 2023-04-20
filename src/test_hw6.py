from utils import *
from num import NUM
from sym import SYM
from data import DATA
from xpln import *
n = 0

def test_the():
    print(the.__repr__())

def test_rand():
    Seed = 1
    t=[]
    for i in range(1,1000+1):
        t.append(rint(0,100,1))
    u=[]
    for i in range(1,1000+1):
        u.append(rint(0,100,1))
    for k,v in enumerate(t):
        assert(v==u[k])

def test_some():
    the['Max'] = 32
    num1 = NUM()
    for i in range(1,1000+1):
        num1.add(i)
    print(num1.has)

def test_num():
    num1, num2 = NUM(), NUM()
    global Seed
    Seed = the['seed']
    for i in range(1,10**3+1):
        num1.add(rand(0,1))
    Seed = the['seed']
    for i in range(1,10**3+1):
        num2.add(rand(0,1)**2)
    m1,m2 = rnd(num1.mid(),1), rnd(num2.mid(),1)
    d1,d2 = rnd(num1.div(),1), rnd(num2.div(),1)
    print(1, m1, d1)
    print(2, m2, d2) 
    return m1 > m2 and .5 == rnd(m1,1)

def test_sym():
    sym = SYM()
    for x in ["a","a","a","a","b","b","c"]:
        sym.add(x)
    print(sym.mid(), rnd(sym.div()))
    return 1.379 == rnd(sym.div())

def no_of_chars_in_file(t):
    global n
    n += len(t)

def test_csv():
    csv(the['file'], no_of_chars_in_file)
    return n > 0

def test_data():
    data = DATA(the['file'])
    col=data.cols.x[2]
    print(col.lo,col.hi, col.mid(),col.div())
    print(data.stats(data.cols.y, 2, 'mid'))

def test_clone():
    data1 = DATA(the['file'])
    data2 = data1.clone(data1.rows)
    print(data1.stats(data1.cols.y, 2, 'mid'))
    print(data2.stats(data2.cols.y, 2, 'mid'))

def test_cliffs():
    assert(False == cliffsDelta( [8,7,6,2,5,8,7,3],[8,7,6,2,5,8,7,3]))
    assert(True  == cliffsDelta( [8,7,6,2,5,8,7,3], [9,9,7,8,10,9,6])) 
    t1,t2=[],[]
    for i in range(1,1000+1):
        t1.append(rand(0,1))
    for i in range(1,1000+1):
        t2.append(rand(0,1)**.5)
    assert(False == cliffsDelta(t1,t1)) 
    assert(True  == cliffsDelta(t1,t2)) 
    diff,j=False,1.0
    while not diff:
        def function(x):
            return x*j
        t3=list(map(function, t1))
        diff=cliffsDelta(t1,t3)
        print(">",rnd(j),diff) 
        j=j*1.025

def test_dist():
    data = DATA(the['file'])
    num  = NUM()
    for row in data.rows:
        num.add(data.dist(row, data.rows[1]))
    print({'lo' : num.lo, 'hi' : num.hi, 'mid' : rnd(num.mid()), 'div' : rnd(num.div())})

def test_half():
    data = DATA(the['file'])
    left,right,A,B,c,_ = data.half() 
    print(len(left),len(right))
    l,r = data.clone(left), data.clone(right)
    print("l",l.stats(l.cols.y, 2, 'mid'))
    print("r",r.stats(r.cols.y, 2, 'mid'))

def test_tree():
    data = DATA(the['file'])
    showTree(data.tree(),"mid",data.cols.y,1)
    return True

def test_sway():
    data = DATA(the['file'])
    best,rest,_ = data.sway()
    print("\nall ", data.stats(data.cols.y, 2, 'mid'))
    print("    ", data.stats(data.cols.y, 2, 'div'))
    print("\nbest",best.stats(best.cols.y, 2, 'mid'))
    print("    ", best.stats(best.cols.y, 2, 'div'))
    print("\nrest", rest.stats(rest.cols.y, 2, 'mid'))
    print("    ", rest.stats(rest.cols.y, 2, 'div'))
    return True

def test_bins():
    global b4
    data = DATA(the['file'])
    best,rest,_ = data.sway()
    print("all","","","",{'best':len(best.rows), 'rest':len(rest.rows)})

def test_xpln():
    data = DATA(the['file'])
    best,rest,evals = data.sway()
    xp = XPLN(best, rest)
    rule,most=  xp.xpln(data,best,rest)
    print("\n-----------\nexplain=", showRule(rule))
    select = selects(rule,data.rows)
    data_select = [s for s in select if s!=None]
    data1= data.clone(data_select)
    print("all               ",data.stats(data.cols.y, 2, 'mid'),data.stats(data.cols.y, 2, 'div'))
    print("sway with",evals,"evals",best.stats(best.cols.y, 2, 'mid'),best.stats(best.cols.y, 2, 'div'))
    print("xpln on",evals,"evals",data1.stats(data1.cols.y, 2, 'mid'),data1.stats(data1.cols.y, 2, 'div'))
    top,_ = data.betters(len(best.rows))
    top = data.clone(top)
    print("sort with",len(data.rows),"evals",top.stats(top.cols.y, 2, 'mid'),top.stats(top.cols.y, 2, 'div'))