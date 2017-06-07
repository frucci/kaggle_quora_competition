#!/bin/sh
# printf "Launching a new model\n"
# python lanciatoreNN.py mynet0 # &> CONTROLLAMI.txt
# sleep 5
# printf "Launching a new model\n"
# python lanciatoreNN.py mynet1 # &>> CONTROLLAMI.txt
# sleep 5
# printf "Launching a new model\n"
# python lanciatoreNN.py mynet2 # &>> CONTROLLAMI.txt
# sleep 5
# printf "Launching a new model\n"
# python lanciatoreNN.py mynet3 # &>> CONTROLLAMI.txt
# sleep 5
# printf "Launching a new model\n"
# python lanciatoreNN.py mynet4 # &>> CONTROLLAMI.txt
# sleep 5
# printf "Launching a new model\n"
# python lanciatoreNN.py mynet5 # &>> CONTROLLAMI.txt

## declare an array variable
declare -a act=("relu") # "sigmoid" "linear" "tanh")
declare -a batch_size=("2048" ) #"4096"  "1024")
declare -a drop=("0.3" ) #"0.2" "0.3" "0.4")

## now loop through the above array
for ac in "${act[@]}"
do
    for bs in "${batch_size[@]}"
    do
        for d in "${drop[@]}"
        do
            printf "\n****************************************\n"
            printf "Training model: $ac $bs $d"
            printf "\n****************************************\n"
            python lanciatoreNN.py mynet_final $ac $bs $d
            printf "\n****************************************\n"
            sleep 5
        done
    done
done


