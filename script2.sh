#!/bin/sh
for seed in 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950
do
    echo ${seed}
    python new_MNIST_main.py -gpuid=1 -seedL=${seed} 
done

