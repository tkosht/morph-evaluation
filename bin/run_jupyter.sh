#!/bin/sh
d=$(cd $(dirname $0) && pwd)
wkdir=$d/../

log_dir="$wkdir/log"
mkdir -p $log_dir

jupyter-notebook --config conf/jupyter_notebook_config.py 2> $log_dir/jupyter.log &
