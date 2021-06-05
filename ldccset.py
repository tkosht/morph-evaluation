import numpy
import pathlib
from dataset import DocRecord


class DatasetLdcc(object):
    def __init__(self, seed=777):
        self.ldccdir = "data/ldcc/text"
        self.rs = numpy.random.RandomState(seed)
        self.dataset = []
        self.labelset = set()
        self.train = None
        self.valid = None

    def load(self):
        docs = []
        labelset = set()
        ldcc_p = pathlib.Path(self.ldccdir)
        for d in sorted(ldcc_p.glob("[a-z]*")):
            labeldir = pathlib.Path(d)
            label = labeldir.name
            labelset.add(label)
            txt_list = labeldir.glob("*-*.txt")
            for txt in sorted(txt_list):
                lines = self._load_lines(txt)
                drec = DocRecord(txt, lines, label)
                docs.append(drec)
        self.dataset = numpy.array(docs)
        self.labelset = labelset
        return self

    def _load_lines(self, txt):
        with open(txt, "r", encoding="UTF-8") as f:
            lines = f.readlines()[2:]   # skip first line as url
        return [l.rstrip() for l in lines]

    def shuffle(self):
        n = len(self.dataset)
        shuffled = self.rs.permutation(range(n))
        self.dataset = self.dataset[shuffled]
        self.train = self.valid = None
        return self

    def split(self, rate=0.7):
        n = len(self.dataset)
        self.n_train = int(n * rate)
        self.n_valid = n - self.n_train
        self.train = self.dataset[:self.n_train]
        self.valid = self.dataset[self.n_train:]
        return self

    def get_data(self, do_split=False):
        if do_split:
            assert self.train is not None
            assert self.valid is not None
            return [dr.doc for dr in self.train], [dr.doc for dr in self.valid]
        return [dr.doc for dr in self.dataset]

    def get_labels(self, do_split=False):
        if do_split:
            assert self.train is not None
            assert self.valid is not None
            return ([dr.label for dr in self.train],
                    [dr.label for dr in self.valid])
        return [dr.label for dr in self.dataset]
