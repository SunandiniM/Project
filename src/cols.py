import re
from sym import SYM
from num import NUM

class COLS:
    def __init__(self, t):
        self.names = t
        self.all = []
        self.x = []
        self.y = []
        self.klass = None

        for col_name in t:
            n = t.index(col_name)
            col_name = col_name.strip()
            if col_name[0].isupper():
                col = NUM(n, col_name)
            else:
                col = SYM(n, col_name)
            self.all.append(col)

            if not col_name[-1] == "X":
                if "!" in col_name:
                    self.klass=col
                self.y.append(col) if re.findall("[!+-]$", col_name) else self.x.append(col)

    def add(self, row):
        for t in [self.x, self.y]:
            for col in t:
                col.add(row.cells[col.at])