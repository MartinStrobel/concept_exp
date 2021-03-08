#!/bin/sh
for seed in 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950
do
    echo ${seed}
    python new_MNIST_main.py -gpuid=0 -seedL=${seed} 
done

