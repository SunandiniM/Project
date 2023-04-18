from row import ROW
from cols import COLS
from utils import *
from operator import itemgetter
from functools import cmp_to_key
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import random

random.seed(the['seed'])

class DATA:
    def __init__(self, src = None, rows = None):
        self.rows = []
        self.cols = None
        if src or rows:
            if isinstance(src, str):
                csv(src, self.add)
            else:
                self.cols = COLS(src.cols.names)
                for row in rows:
                    self.add(row)

    def add(self, t):
        if self.cols:
            t = t if isinstance(t, ROW) else ROW(t)
            self.rows.append(t)
            self.cols.add(t)
        else:
            self.cols=COLS(t)
    
    def stats(self, cols = None, nPlaces = 2, what = 'mid'):
        stats_dict = dict(sorted({col.txt: rnd(getattr(col, what)(), nPlaces) for col in cols or self.cols.y}.items()))
        stats_dict["N"] = len(self.rows)
        return stats_dict
    
    def dist(self, row1, row2, cols = None):
        n,d = 0,0
        for col in cols or self.cols.x:
            n = n + 1
            d = d + col.dist(row1.cells[col.at], row2.cells[col.at])**the['p']
        return (d/n)**(1/the['p'])

    def clone(data, ts={}):
        data1 = DATA()
        data1.add(data.cols.names)
        for _, t in enumerate(ts or {}):
            data1.add(t)
        return data1

    def half(self, rows = None, cols = None, above = None):
        def gap(row1,row2): 
            return self.dist(row1,row2,cols)
        def project(row):
            return {'row' : row, 'dist' : cosine(gap(row,A), gap(row,B), c)}
        rows = rows or self.rows
        some = many(rows,the['Halves'])
        A    = above if above and the['Reuse'] else any(some)
        tmp = sorted([{'row': r, 'dist': gap(r, A)} for r in some], key=lambda x: x['dist'])
        far = tmp[int((len(tmp) - 1) * the['Far'])]
        B    = far['row']
        c    = far['dist']
        left, right = [], []
        for n,tmp in enumerate(sorted(map(project, rows), key=lambda x: x['dist'])):
            if (n + 1) <= (len(rows) / 2):
                left.append(tmp["row"])
            else:
                right.append(tmp["row"])
        evals = 1 if the['Reuse'] and above else 2
        return left, right, A, B, c, evals
    
    def better(self, rows1, rows2, s1=0, s2=0, ys=None, x=0, y=0):
        if isinstance(rows1, ROW):
            rows1 = [rows1]
            rows2 = [rows2]
        if not ys:
            ys = self.cols.y
        for col in ys:
            for row1, row2 in zip(rows1, rows2):
                x = col.norm(row1.cells[col.at])
                y = col.norm(row2.cells[col.at])
                s1 = s1 - math.exp(col.w * (x - y) / len(ys))
                s2 = s2 - math.exp(col.w * (y - x) / len(ys))
        return s1 / len(ys) < s2 / len(ys)
    
    def bdom(self, rows1, rows2, ys=None):
        if isinstance(rows1, ROW):
            rows1 = [rows1]
            rows2 = [rows2]
        if not ys:
            ys = self.cols.y
        
        dominates = False
        for col in ys:
            for row1, row2 in zip(rows1, rows2):
                x = col.norm(row1.cells[col.at]) * col.w * -1
                y = col.norm(row2.cells[col.at]) * col.w * -1
                if x > y:
                    return False
                elif x < y:
                    dominates = True
        return dominates

    def better_bdom(self, row1, row2, ys=None):
        row1_bdom = self.bdom(row1, row2, ys=ys)
        row2_bdom = self.bdom(row2, row1, ys=ys)
        if row1_bdom and not row2_bdom:
            return True
        else:
            return False
        
    def better_hypervolume(self, row1, row2):
        s1, s2, ys = 0, 0, self.cols.y
        data = [[], []]
        ref_point = []
        for col in ys:
            x = col.norm(row1.cells[col.at])
            y = col.norm(row2.cells[col.at])
            if '-' in col.txt:
                x = -x
                y = -y
                ref_point.append(1)
            else:
                ref_point.append(2)
            data[0].append(x)
            data[1].append(y)
        if len(ref_point) < 2:
            return data[0] > data[1]
        # print(data)
        # print(ref_point)

        hv = hypervolume(data)
        output = hv.contributions(ref_point)
        hv1, hv2 = output[0], output[1]

        return hv1 < hv2
    
    def tree(self, rows = None , min = None, cols = None, above = None):
        rows = rows or self.rows
        min  = min or len(rows)**the['min']
        cols = cols or self.cols.x
        node = { 'data' : self.clone(rows) }
        if len(rows) >= 2*min:
            left, right, node['A'], node['B'], _, _ = self.half(rows,cols,above)
            node['left']  = self.tree(left,  min, cols, node['A'])
            node['right'] = self.tree(right, min, cols, node['B'])
        return node
    
    def sway(self, algo = 'half', better = 'zitler'):
        data = self
        def worker(rows, worse, evals0 = None, above = None):
            if len(rows) <= len(data.rows)**the['min']: 
                return rows, many(worse, the['rest']*len(rows)), evals0
            else:
                if algo == 'half':
                    l,r,A,B,c,evals = self.half(rows, None, above)
                elif algo == 'kmeans':
                    l,r,A,B,evals = self.kmeans(rows)
                elif algo == 'agglomerative_clustering':
                    l,r,A,B,evals = self.agglomerative_clustering(rows)
                elif algo == 'dbscan':
                    l,r,A,B,evals = self.dbscan(rows)
                elif algo == 'pca':
                    l,r,A,B,evals = self.pca(rows)
                
                if better == 'zitler':
                    if self.better(B,A):
                        l,r,A,B = r,l,B,A
                elif better == 'bdom':
                    if self.better_bdom(B,A):
                        l,r,A,B = r,l,B,A

                for row in r:
                    worse.append(row)
                return worker(l,worse,evals+evals0,A)
        best,rest,evals = worker(data.rows,[],0)
        return DATA.clone(self, best), DATA.clone(self, rest), evals
    
    def betters(self,n):
        key = cmp_to_key(lambda row1, row2: -1 if self.better(row1, row2) else 1)
        tmp = sorted(self.rows, key = key)
        if n is None:
            return tmp
        else:
            return tmp[1:n], tmp[n+1:]
    
    def kmeans(self, rows=None):
        left = []
        right = []
        A = None
        B = None
        
        def min_dist(center, row, A):
            if not A:
                A = row
            if self.dist(A, center) > self.dist(A, row):
                return row
            else:
                return A
    
        if not rows:
            rows = self.rows
        row_set = np.array([r.cells for r in rows])
        kmeans = KMeans(n_clusters=2, random_state=the['seed'], n_init=10)
        kmeans.fit(row_set)
        left_cluster = ROW(kmeans.cluster_centers_[0])
        right_cluster = ROW(kmeans.cluster_centers_[1])

        for key, value in enumerate(kmeans.labels_):
            if value == 0:
                A = min_dist(left_cluster, rows[key], A)
                left.append(rows[key])
            else:
                B = min_dist(right_cluster, rows[key], B)
                right.append(rows[key])

        return left, right, A, B, 1
    
    def agglomerative_clustering(self, rows=None):
        left = []
        right = []

        if not rows:
            rows = self.rows
        row_set = np.array([r.cells for r in rows])
        agg_clust = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
        agg_clust.fit(row_set)

        for key, value in enumerate(agg_clust.labels_):
            if value == 0:
                left.append(rows[key])
            else:
                right.append(rows[key])
        return left, right, random.choices(left, k=10), random.choices(right, k=10), 1
    
    def dbscan(self, rows=None):
        left = []
        right = []

        if not rows:
            rows = self.rows
        row_set = np.array([r.cells for r in rows])
        db = DBSCAN(eps = 3, min_samples = 2)
        db.fit(row_set)

        for key, value in enumerate(db.labels_):
            if value == 0:
                left.append(rows[key])
            else:
                right.append(rows[key])
        return left, right, random.choices(left, k=10), random.choices(right, k=10), db.n_features_in_
    
    def pca(self, rows=None, cols=None, above=None):
        if not rows:
            rows = self.rows
        row_set = np.array([r.cells for r in rows])
        pca = PCA(n_components=1)
        pcs = pca.fit_transform(row_set)
        result = []
        for i in sorted(enumerate(rows), key=lambda x: pcs[x[0]]):
            result.append(i[1])
        n = len(result)
        left = result[:n//2]
        right = result[n//2:]
        return left, right, random.choices(left, k=10), random.choices(right, k=10), 1