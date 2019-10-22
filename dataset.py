import sys


class DocRecord(object):
    def __init__(self, url, doc, label):
        self.url = url
        self.doc = doc
        self.label = label

    def __getstate__(self):
        doc = []
        for s in self.doc:  # for self.doc is a generator
            doc.append(s)
        state = {
            "url": self.url,
            "doc": doc,
            "label": self.label,
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Dataset(object):
    def load(self):
        # return self
        raise NotImplementedError(sys._getframe().f_code.co_name)

    def shuffle(self):
        # return self
        raise NotImplementedError(sys._getframe().f_code.co_name)

    def split(self, train_rate=0.7):
        # return self
        raise NotImplementedError(sys._getframe().f_code.co_name)

    def get_data(self, do_split=True):
        # return [drec.doc for drec in self.dataset]
        raise NotImplementedError(sys._getframe().f_code.co_name)
        
    def get_labels(self, do_split=True):
        # return [drec.label for drec in self.dataset]
        raise NotImplementedError(sys._getframe().f_code.co_name)
