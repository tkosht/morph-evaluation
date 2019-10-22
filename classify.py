import pathlib
from datetime import datetime

try:
    import MeCab
except Exception:
    MeCab = None

try:
    import janome.tokenizer
    import janome.analyzer
    import janome.charfilter
    import janome.tokenfilter
except Exception:
    janome = None

try:
    import sudachipy.dictionary
    import sudachipy.tokenizer
except Exception:
    sudachipy = None

try:
    import nagisa
except Exception:
    nagisa = None

try:
    import sentencepiece as spm
except Exception:
    spm = None

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
        self.taggerstr = f"-O chasen -d {self.dicdir}"
        self.tokenizer = MeCab.Tagger(self.taggerstr)

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

    def __getstate__(self):
        state = {
            "dicdir": self.dicdir,
            "taggerstr": self.taggerstr,
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = MeCab.Tagger(self.taggerstr)


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

    def __getstate__(self):
        state = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class JpTokenizerSudachi(JpTokenizer):
    def __init__(self):
        self.toker = sudachipy.dictionary.Dictionary().create()
        # self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.B
        self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.C

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


class JpTokenizerSentencePiece(JpTokenizer):
    def __init__(self,
                 input_txt="wk/sp.txt",
                 model_prefix="wk/sp",
                 vocab_size=2000):

        self.input_txt = input_txt
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_file = f"{self.model_prefix}.model"
        self.sp = None

    def fit(self, X, y, **kwargs):
        pathlib.Path(self.input_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(self.input_txt, "w", encoding="utf-8") as f:
            for doc in X:
                f.writelines(doc)
        param_str = f"""--input={self.input_txt}
                     --model_prefix={self.model_prefix}
                     --vocab_size={self.vocab_size}
                     """
        param_str = param_str.replace("\n", "")
        spm.SentencePieceTrainer.train(param_str)
        self._load_model()
        return self

    def _load_model(self):
        sp = spm.SentencePieceProcessor()
        sp.Load(self.model_file)
        self.sp = sp

    def tokenize(self, line):
        pieces = self.sp.encode_as_pieces(line)
        return pieces

    def __getstate__(self):
        state = {
            "input_txt": self.input_txt,
            "model_prefix": self.model_prefix,
            "vocab_size": self.vocab_size,
            "model_file": self.model_file,
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model()


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


def build_pipleline_simple(tokener):
    tfidf = TfidfVectorizer(
                tokenizer=ident_tokener,
                lowercase=False
                )

    embedders = [
        ("pca", PCA(n_components=32)),
        ("identity", Transer()),    # means tfidf to tfidf
    ]

    lgbmclf = lightgbm.LGBMClassifier(
                objective="softmax",
                num_class=len(dataset.labelset),
                importance_type="gain",
                )

    pipe = Pipeline(steps=[
        ("tokenizer", tokener),
        ("vectorizer", tfidf),
        ("to_dence", SparsetoDense()),
        ("embedder", FeatureUnion(embedders)),
        ("classifier", lgbmclf),
    ])

    return pipe


def build_pipleline_with_doc2vec(tokener):
    tfidf = TfidfVectorizer(
                tokenizer=ident_tokener,
                lowercase=False
                )

    embedders = [
        ("pca", PCA(n_components=32)),
        ("identity", Transer()),    # means tfidf to tfidf
    ]

    pipe_embedder_1 = Pipeline(steps=[
        ("vectorizer", tfidf),
        ("to_dence", SparsetoDense()),
        ("embedder", FeatureUnion(embedders)),
    ])
    pipe_embedder_2 = Pipeline(steps = [
        ("doctagger", TagDocMaker()),
        ("doc2vec", Doc2Vectorizer(n_components=128, min_count=1)),
    ])
    pipe_embeds = [
        ("pipe1", pipe_embedder_1),
        ("pipe2", pipe_embedder_2),
        ]

    lgbmclf = lightgbm.LGBMClassifier(
                objective="softmax",
                num_class=len(dataset.labelset),
                importance_type="gain",
                )

    pipe = Pipeline(steps=[
        ("tokenizer", tokener),
        ("embedders", FeatureUnion(pipe_embeds)),
        ("classifier", lgbmclf),
    ])

    return pipe


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--dataset',
                        choices=['ldcc', 'aozora'],
                        default="aozora",
                        help='string for the dataset name (default: "aozora")')
    parser.add_argument('--iter',
                        type=int,
                        default=3,
                        help='positive integer of the iteration '
                             'for train and validation (default: 10)')
    args = parser.parse_args()
    return args


def get_now():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def print_log(*params):
    print(get_now(), *params)


if __name__ == '__main__':
    import sys
    import time
    import argparse
    import lightgbm
    import joblib
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA   # , KernelPCA
    from sklearn.metrics import accuracy_score
    from ldccset import DatasetLdcc
    from aozoraset import DatasetAozora

    args = get_args()

    # load dataset
    data_file = f"model/{args.dataset}set.gz"
    if pathlib.Path(data_file).exists():
        dataset = joblib.load(data_file)
    else:
        print_log(f"loading dataset {args.dataset} ...")
        dataset_class_dic = dict(
            ldcc=DatasetLdcc,
            aozora=DatasetAozora,
            )

        dataset_class = dataset_class_dic[args.dataset]
        dataset = dataset_class()
        dataset.load()
        print_log(f"loading dataset {args.dataset} ... Done.")

        print(f"Saving dataset ... [{data_file}]")
        pathlib.Path("./model").mkdir(parents=True, exist_ok=True)
        joblib.dump(dataset, data_file, compress=("gzip", 3))

    # setup tokenizers
    tokenizers = []
    if MeCab is not None:
        tokenizers.append(JpTokenizerMeCab())
#     if janome is not None:
#         tokenizers.append(JpTokenizerJanome())
#     if sudachipy is not None:
#         tokenizers.append(JpTokenizerSudachi())
#     if nagisa is not None:
#         tokenizers.append(JpTokenizerNagisa())
    if spm is not None:
        tokenizers.append(JpTokenizerSentencePiece(vocab_size=5000))

    # loop to train and validation
    print("tokenizer, train_acc, valid_acc, elapsed_time, cpu_time")
    for _ in range(args.iter):
        dataset.shuffle().split()
        X_train, X_valid = dataset.get_data(do_split=True)
        y_train, y_valid = dataset.get_labels(do_split=True)

        for tokener in tokenizers:
            print(tokener.__class__.__name__,
                  "Processing ...", file=sys.stderr)

            build_pipleline = build_pipleline_simple
            # build_pipleline = build_pipleline_with_doc2vec
            pipe = build_pipleline(tokener)

            tps = time.perf_counter()
            tcs = time.process_time()

            pipe.fit(X_train, y_train)

            # predict trainset
            p = pipe.predict(X_train)
            train_acc = accuracy_score(y_train, p)

            # predict validset
            p = pipe.predict(X_valid)
            valid_acc = accuracy_score(y_valid, p)

            tce = time.process_time()
            tpe = time.perf_counter()

            elapsed_time = tpe - tps
            cpu_time = tce - tcs

            print(f"{tokener.__class__.__name__}, "
                  f"{train_acc}, {valid_acc}, "
                  f"{elapsed_time}, {cpu_time}")

        # save model
        print(f"Saving model for {tokener.__class__.__name__.lower()} ...")
        pipe_file = f"model/pipe-{tokener.__class__.__name__.lower()}.gz"
        joblib.dump(pipe, pipe_file, compress=("gzip", 3))

        data_file = f"model/{args.dataset}set.gz"
        print(f"Saving dataset ... [{data_file}]")
        joblib.dump(dataset, data_file, compress=("gzip", 3))

        print(tokener.__class__.__name__,
              "Done.", file=sys.stderr)
