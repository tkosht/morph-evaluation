import io
import gzip
import requests
import pandas
import numpy
import lxml.html
import warnings
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from memory_profiler import profile

from dataset import Dataset, DocRecord


class DatasetAozora(object):
    url = "http://aozora-word.hahasoha.net/aozora_word_list_utf8.csv.gz"
    n_threshold = 300

    def __init__(self, seed=777):
        self.seed = seed
        self.rs = numpy.random.RandomState(seed)
        self.list_df = None
        self.datalist_df = None
        self.dataset = None
        self.labelset = None
        self.train = None
        self.valid = None
        
    def _get_list(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            gzip_file = io.BytesIO(response.content)
            with gzip.open(gzip_file, 'rt') as f:
                self.list_df = pandas.read_csv(f, header=0, encoding="utf-8")
        return self

    def _make_datalist(self):
        count_df = self.list_df.groupby("分類番号").count()["作品id"]
        categories = set(count_df[count_df > self.n_threshold].index)
        columns = ["作品id", "作品名", "分類番号", "XHTML/HTMLファイルURL", "XHTML/HTMLファイル符号化方式"]
        datalist_df = pandas.DataFrame([])
        for ctg in categories:
            q = f"分類番号 == '{ctg}'"
            cat_df = self.list_df.query(q).sample(self.n_threshold, random_state=self.rs)[columns]
            datalist_df = pandas.concat([datalist_df, cat_df], axis=0)
        datalist_df = datalist_df.reset_index().drop("index", axis=1)
        self.datalist_df = datalist_df
        self.labelset = numpy.array(list(categories))
        return self
    
    def _load_list(self):
        self._get_list()
        self._make_datalist()
        
    @profile
    def load(self):
        self._load_list()
        
        dataset = []
        for idx, rec in tqdm(self.datalist_df.iterrows()):
            try:
                url = rec["XHTML/HTMLファイルURL"]
                label = rec["分類番号"]
                content = self._get_content(url)
                lines = self._to_lines(content)
                drec = DocRecord(url, lines, label)
                dataset.append(drec)
            except Exception as e:
                warnings.warn(f"Couldn't get the html via http[{e}], url: {url}")
                continue
        self.dataset = numpy.array(dataset)
        return self
        
    def _get_content(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"response code: {response.status_code}")
        html = lxml.html.fromstring(response.content)
        main_text = html.xpath("//div[@class='main_text']")
        content = main_text[0].text_content()
        return content
                
    def _to_lines(self, content):
        return [l + "\n" for l in content.split("\n")]
        # eos = "。\n"
        # _content = content.replace("\n", "")
        # _content = _content.replace("。」", "__rbracket___")
        # sentences = _content.split("。")
        # for s in sentences:
        #     s = s.strip()
        #     s = s.replace("__rbracket___", "。」")
        #     if s == "":
        #         continue
        #     yield s + eos

    def shuffle(self):
        numpy.random.shuffle(self.dataset)
        self.train = self.valid = None
        return self

    def split(self, train_rate=0.7):
        n = len(self.dataset)
        self.n_train = int(n * train_rate)
        self.n_valid = n - self.n_train
        self.train = self.dataset[:self.n_train]
        self.valid = self.dataset[self.n_train:]
        return self

    def get_data(self, do_split=True):
        if do_split:
            assert self.train is not None
            assert self.valid is not None
            return [drec.doc for drec in self.train], [drec.doc for drec in self.valid]
        return [drec.doc for drec in self.dataset]
        
    def get_labels(self, do_split=True):
        if do_split:
            assert self.train is not None
            assert self.valid is not None
            return [drec.label for drec in self.train], [drec.label for drec in self.valid]
        return [drec.label for drec in self.dataset]
