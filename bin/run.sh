#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

echo "`date +'%Y/%m/%d %T'` - Start" | tee .run.log
unbuffer python classify.py $* | tee -a .run.log
echo "`date +'%Y/%m/%d %T'` - End" | tee -a .run.log
