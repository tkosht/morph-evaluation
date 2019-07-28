import numpy
import pathlib
try: import MeCab
except: MeCab = None
try:
    import janome.tokenizer
    import janome.analyzer
    import janome.charfilter
    import janome.tokenfilter
except: janome = None
try:
    import sudachipy.dictionary
    import sudachipy.tokenizer
except: sudachipy = None
try: import nagisa
except: nagisa = None
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class DocRecord(object):
    def __init__(self, fname, doc, label):
        self.fname = fname
        self.doc = doc    # may be an array of lines
        self.label = label


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
            lines = f.readlines()[2:]
        return lines

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

    def get_data(self, with_split=False):
        if with_split:
            assert self.train is not None
            assert self.valid is not None
            return [dr.doc for dr in self.train], [dr.doc for dr in self.valid]
        return [dr.doc for dr in self.dataset]

    def get_labels(self, with_split=False):
        if with_split:
            assert self.train is not None
            assert self.valid is not None
            return ([dr.label for dr in self.train],
                    [dr.label for dr in self.valid])
        return [dr.label for dr in self.dataset]


class Transer(object):
    def transform(self, X, **kwargs):
        return X

    def fit(self, X, y, **kwargs):
        return self


g_stop_poses = ["BOS/EOS", "助詞", "助動詞", "接続詞", "記号", "補助記号", "未知語"]


class JpTokenizer(Transer):
    def transform(self, X, **kwargs):
        docs = []
        for lines in X:
            doc = []
            for line in lines:
                sentence = self.tokenize(line)
                doc.extend(sentence)
            docs.append(doc)
        return docs

    def tokenize(self, line: str) -> list:
        # return line.split(" ") for example
        raise NotImplementedError("tokenize()")


class JpTokenizerMeCab(JpTokenizer):
    def __init__(self):
        self.dicdir = ("/usr/lib/x86_64-linux-gnu/mecab/dic"
                       "/mecab-ipadic-neologd")
        self.tokenizer = MeCab.Tagger(f"-O chasen -d {self.dicdir}")

    def tokenize(self, line):
        sentence = []
        parsed = self.tokenizer.parse(line)
        splitted = [l.split("\t") for l in parsed.split("\n")]
        for s in splitted:
            if len(s) == 1:     # may be "EOS"
                break
            word = s[2]         # original form
            pos = s[3].split("-")[0]
            if pos not in g_stop_poses:
                sentence.append(word)
        return sentence


class JpTokenizerJanome(JpTokenizer):
    def __init__(self):
        char_filters = [janome.charfilter.UnicodeNormalizeCharFilter()]
        tokenizer = janome.tokenizer.Tokenizer()
        token_filters = [janome.tokenfilter.POSStopFilter(g_stop_poses)]
        self.aly = janome.analyzer.Analyzer(
                    char_filters, tokenizer, token_filters)

    def tokenize(self, line):
        sentence = []
        for token in self.aly.analyze(line):
            sentence.append(token.base_form)
        return sentence


class JpTokenizerSudachi(JpTokenizer):
    def __init__(self):
        self.toker = sudachipy.dictionary.Dictionary().create()
        self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.B
        # self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.C

    def tokenize(self, line):
        sentence = []
        for token in self.toker.tokenize(line, self.mode):
            if token.part_of_speech()[0] not in g_stop_poses:
                sentence.append(token.dictionary_form())
        return sentence


class JpTokenizerNagisa(JpTokenizer):
    def tokenize(self, line):
        tagged = nagisa.filter(line, filter_postags=g_stop_poses)
        return tagged.words


class SparsetoDense(Transer):
    def transform(self, X, **kwargs):
        return X.toarray()


class TagDocMaker(Transer):
    def transform(self, X, **kwargs):
        return [TaggedDocument(words=sentences, tags=[n])
                for n, sentences in enumerate(X)]


class Doc2Vectorizer(Transer):
    def __init__(self, n_components, window=7, min_count=1, workers=3):
        super().__init__()
        self.model = None
        self.vector_size = n_components
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def transform(self, X, **kwargs):
        embedded = []
        for tagdoc in X:
            v = self.model.infer_vector(tagdoc.words)
            embedded.append(v)
        return embedded

    def fit(self, X, y, **kwargs):
        assert isinstance(X, list)
        assert isinstance(X[0], TaggedDocument)
        params = dict(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            dm=0,
            dbow_words=0,
            negative=0,
            hs=1,
        )
        self.model = Doc2Vec(X, **params)
        return self


def ident_tokener(sentence):
    return sentence


if __name__ == '__main__':
    import sys
    import time
    import lightgbm
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA   # , KernelPCA
    from sklearn.metrics import accuracy_score

    ldccset = DatasetLdcc()
    ldccset.load()

    tokenizers = []
    if MeCab is not None:
        tokenizers.append(JpTokenizerMeCab())
    if janome is not None:
        tokenizers.append(JpTokenizerJanome())
    if sudachipy is not None:
        tokenizers.append(JpTokenizerSudachi())
    if nagisa is not None:
        tokenizers.append(JpTokenizerNagisa())

    print("tokenizer, train_acc, valid_acc, elapsed_time, cpu_time")
    for _ in range(10):
        ldccset.shuffle().split()
        X_train, X_valid = ldccset.get_data(with_split=True)
        y_train, y_valid = ldccset.get_labels(with_split=True)

        for tokener in tokenizers:
            print(tokener.__class__.__name__,
                  "Processing ...", file=sys.stderr)

            embedders = [
                ("pca", PCA(n_components=32)),
                ("identity", Transer()),    # means tfidf to tfidf
            ]

            tfidf = TfidfVectorizer(
                        tokenizer=ident_tokener,
                        lowercase=False
                        )
            lgbmclf = lightgbm.LGBMClassifier(
                        objective="softmax",
                        num_class=len(ldccset.labelset)
                        )
            pipe = Pipeline(steps=[
                ("tokenizer", tokener),
                ("vectorizer", tfidf),
                ("to_dence", SparsetoDense()),
                ("embedder", FeatureUnion(embedders)),
                ("classifier", lgbmclf),
            ])

            tps = time.perf_counter()
            tcs = time.process_time()

            pipe.fit(X_train, y_train)
            p = pipe.predict(X_train)
            train_acc = accuracy_score(y_train, p)
            p = pipe.predict(X_valid)
            valid_acc = accuracy_score(y_valid, p)

            tce = time.process_time()
            tpe = time.perf_counter()

            elapsed_time = tpe - tps
            cpu_time = tce - tcs

            print(f"{tokener.__class__.__name__}, "
                  f"{train_acc}, {valid_acc}, "
                  f"{elapsed_time}, {cpu_time}")
            print(tokener.__class__.__name__,
                  "Done.", file=sys.stderr)
