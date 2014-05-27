#!/bin/bash
y=128

for i in 1 2 4 8 16 32 64 128
do
   sed -i "s/#define NEVTS $y/#define NEVTS $i/g" svtsim_functions.h 
   cat svtsim_functions.h | grep "define NEVTS"
   make clean && make
   ./svt_gpu -t -i ebjt0p_fromp36_hboutForGPU -l 100 
   y=$i
done
