#!/bin/sh
set -e
d=$(cd $(dirname $0) && pwd)
cd $d/../

mkdir -p sudachi
cd sudachi
wget https://object-storage.tyo2.conoha.io/v1/nc_2520839e1f9641b08211a5c85243124a/sudachi/SudachiDict_full-20190718.tar.gz
pip install SudachiDict_full-20190718.tar.gz
sudachipy link -t full  # after this, you can create Dictionaly() object
