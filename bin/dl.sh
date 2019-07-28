#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

ldcc_dir="data/ldcc"
rm -rf $ldcc_dir
mkdir -p $ldcc_dir
cd $ldcc_dir

url=https://www.rondhuit.com/download/ldcc-20140209.tar.gz
bn=$(basename $url)
wget $url
tar xzf $bn
