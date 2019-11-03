# morph-evaluation
An evaluation repository for the Japanese Morphological Analysis Tools,
like MeCab, Janome, SudachiPy, nagisa, SentencePiece.
but, by now, SudachiPy and nagisa is too slow to process as using in business,
so just commented out them.

The using datasets in japenese are the `ldcc`
([libdoor news corpus](https://www.rondhuit.com/download.html)) dataset
and the `aozora` ([青空文庫形態素解析データ集](http://aozora-word.hahasoha.net/download.html)) dataset.

The evaluation task is classification task:
- 5 categories (filtered by program) in the `aozora` dataset
- 9 categories in the `ldcc` dataset

you can view the evaluation result and a simple analysis in the [notebook](notebook/report-evaluation.ipynb), please check it out.


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

# preparation
if you want to use `ldcc` dataset, download it

```bash
sh bin/dl.sh
```

# run

if you use `aozora` dataset (default), you may just run `classify.py`

```bash
python classify.py
```

if you use `ldcc` dataset, you may specify the option as `--dataset=ldcc`

```bash
python classify.py --dataset=ldcc
```

you can check the help message

```bash
python classify.py --help
```

## output example

```bash
[dynet] random seed: 1234
[dynet] allocating memory: 32MB
[dynet] memory allocation done.
'pattern' package not found; tag filters are not available for English
tokenizer, train_acc, valid_acc, elapsed_time, cpu_time
JpTokenizerMeCab Processing ...
JpTokenizerMeCab, 1.0, 0.7955555555555556, 68.88072609300434, 230.663923129
JpTokenizerSentencePiece Processing ...
sentencepiece_trainer.cc(116) LOG(INFO) Running command: --input=wk/sp.txt                     --model_prefix=wk/sp                     --vocab_size=5000
sentencepiece_trainer.cc(49) LOG(INFO) Starts training with :
:
```
