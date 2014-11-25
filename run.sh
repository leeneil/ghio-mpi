#!/bin/sh
#BSUB -a openmpi
#BSUB -n 48
#BSUB -J ghio
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R 'span[ptile=48]'
##BSUB -R "ipathavail==0" 

mpirun.lsf -np $LSB_DJOB_NUMPROC ./ghio2d2 sp8_nov_ms1_3a.dat | tee sp8_nov_ms1_3a.out 
