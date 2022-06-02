#!/bin/bash

for batchsize in 4 8 16 32 64 128
do
    for lr in 0.01 0.001 0.0001
    do
        for early_stopping in 25
        do
            for reduce_lr in 2000
            do
                for epoch in 2000
                do
                    for local_epoch in 1 #2 4 8 16
                    do
                        python3 -m run -clients 3 --name "noniidclient3" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input3/' --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 3 --name "noniidclient3" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input3/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 3 --name "noniidclient3" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input3/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 1 --name "local" -b $batchsize --lr $lr -e $epoch -g True -input '/data/global/' --local_epoch $local_epoch --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 1 --name "local" -b $batchsize --lr $lr -e $epoch -g True -input '/data/global/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 1 --name "local" -b $batchsize --lr $lr -e $epoch -g True -input '/data/global/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 2 --name "client2" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input/' --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 2 --name "client2" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 2 --name "client2" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 3 --name "client3" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input3/' --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 3 --name "client3" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input3/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 3 --name "client3" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input3/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 5 --name "client5" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input5/' --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 5 --name "client5" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input5/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 5 --name "client5" -b $batchsize --lr $lr -e $epoch -g True -input '/data/input5/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 2 --name "clientniid2" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input2/' --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 2 --name "clientniid2" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input2/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 2 --name "clientniid2" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input2/'  --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                        python3 -m run -clients 6 --name "clientniid6" -b $batchsize --lr $lr -e $epoch -g True -input '/data/noniid_input6/' --local_epoch $local_epoch  --param_factor 1 --reduce_lr 2000
                    done
                done
            done
        done
    done
done