# morph-evaluation
An evaluation repository for the Japanese Morphological Analysis Tools,
like MeCab, Janome, SudachiPy, nagisa.
the using dataset in japenese is the `ldcc` dataset
([libdoor news corpus](https://www.rondhuit.com/download.html)).
the evaluation task is a classification of the 9 categories in `ldcc` dataset


---

# Install

## install python libraries

```bash
sh bin/pip.sh
```

## Install Dictionaries

### for MeCab

```
sh bin/install_mecabdic.sh
```

### for SudachiPy

```
bin/install_sudachidic.sh
```

---

# run

```bash
python classify_ldcc.py
```
## output example

```bash
[dynet] random seed: 1234
[dynet] allocating memory: 32MB
[dynet] memory allocation done.
'pattern' package not found; tag filters are not available for English
tokenizer, train_acc, valid_acc, elapsed_time, cpu_time
JpTokenizerMeCab Processing ...
JpTokenizerMeCab, 1.0, 0.9434644957033017, 62.64325175799604, 300.346383533
JpTokenizerMeCab Done.
JpTokenizerJanome Processing ...
:
```
