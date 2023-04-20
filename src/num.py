import math
from utils import *
from config import the

class NUM:
    def __init__(self, at=0, txt="", t = None):
        self.at = at
        self.txt = txt
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.sd = 0
        self.lo = float('inf')
        self.hi = float('-inf')
        self.w = -1 if self.txt.endswith("-") else 1
        self.has = {}

        if t:
            for n in t:
                self.add(n)
        else:
            self.add(0)

    def add(self, n):
        if n != '?':
            self.n += 1
            self.lo, self.hi = min(n, self.lo), max(n, self.hi)

            all = len(self.has)

            pos = all + 1 if all < the["Max"] else rint(1, all) if rand() < the["Max"] / self.n else 0

            if pos:
                self.has[pos] = n
                self.ok = False

            d = n - self.mu
            self.mu = self.mu + d/self.n
            self.m2 = self.m2 + d*(n-self.mu)
            self.sd = 0 if self.n<2 else (self.m2/(self.n - 1))**.5

    def mid(self):
        return per(self.vals(), .5)

    def div(self):
        return (per(self.vals(), .9) - per(self.vals(), .1)) / 2.58
    
    def vals(self):
        return list(dict(sorted(self.has.items(), key=lambda x: x[1])).values())

    def rnd(self, x, n):
        if x == '?':
            return x
        else:
            return rnd(x, n)

    def norm(self, n):
        return n if n == '?' else (n - self.lo) / (self.hi - self.lo + 1e-32)

    def dist(col, x, y):
        if x == "?" and y == "?":
            return 1
        if type(col) is SYM:
            return 0 if x == y else 1
        x, y = col.norm(x), col.norm(y)
        if x == "?":
            x = 1 if y < 0.5 else 1
        if y == "?":
            y = 1 if x < 0.5 else 1
        return abs(x - y)