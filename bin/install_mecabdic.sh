#!/bin/sh
set -e
d=$(cd $(dirname $0) && pwd)
cd $d/../

mkdir -p mecab
cd mecab
git clone https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd && sudo bin/install-mecab-ipadic-neologd -a
