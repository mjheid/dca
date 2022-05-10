#!/bin/bash

for batchsize in 8 16 32 64
do
    for lr in 0.1 0.01 0.001 0.0001
    do
        for early_stopping in 15 25 50 100
        do
            for reduce_lr in 10 20 40 80
            do
                for epoch in 2000
                do 
                    python3 -m run -clients 2 --name "client2t" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input/'
                    python3 -m run -clients 2 --name "client2t" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input/'
                    python3 -m run -clients 2 --name "client2t" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input/'
                done
            done
        done
    done
done