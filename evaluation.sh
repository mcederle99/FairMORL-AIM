#!/bin/bash

# Script to reproduce results
for ((s=1;s<11;s+=1))
do
  python evaluation.py \
  --seed $s \
  --file_name continuous_smart
done
