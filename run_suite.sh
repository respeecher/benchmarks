#!/bin/bash

matmul () {
    SIZE=$1
    ITER=$2
    echo -e "\n\nmatmul@${SIZE}:" | tee -a $LOGFILE
    python super_burner.py --framework pytorch \
                           --task matmul \
                           --matrix-size $SIZE \
                           --iterations $ITER | tee -a $LOGFILE
}

rnn () {
    SEQ=$1
    ITER=$2
    echo -e "\n\nrnn@seq=${SEQ}:" | tee -a $LOGFILE
    python super_burner.py --framework pytorch \
                           --task rnn \
                           --input-size $SEQ \
                           --iterations $ITER | tee -a $LOGFILE
}

conv2d () {
    INP=$1
    KER=$2
    ITER=$3
    echo -e "\n\nconv2d@inp=${INP};ker=${KER}:" | tee -a $LOGFILE
    python super_burner.py --framework pytorch \
                           --task conv2d \
                           --input-size-2d $INP \
                           --kernel-size $KER \
                           --iterations $ITER | tee -a $LOGFILE
}

lstm () {
    SEQ=$1
    ITER=$2
    echo -e "\n\nlstm@${SEQ}:" | tee -a $LOGFILE
    python super_burner.py --framework pytorch \
                           --task lstm \
                           --input-size $SEQ \
                           --iterations $ITER | tee -a $LOGFILE
}


LOGFILE=$1
GPU_DEVICE=$2

if [ ! -z "$GPU_DEVICE" ]
then
    echo "##### RUNNING TESTS ON GPU (DEVICE ${GPU_DEVICE}) #########"
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

    matmul 128 20000
    matmul 512 1000
    matmul 2048 50

    rnn 30 20000
    rnn 300 20000
    rnn 3000 20000

    conv2d 30 3 7000
    conv2d 100 10 500
    conv2d 100 50 1000

    lstm 300 10000
    lstm 3000 10000
    lstm 30000 2000
else
    echo "WARNING: Skipping GPU tests. If you want to run GPU tests,
    specify the GPU device id as a second argument to this script."
fi


echo "##### RUNNING TESTS ON CPU ################################"
export CUDA_VISIBLE_DEVICES=

matmul 128 1500
matmul 512 100
matmul 2048 10

rnn 30 6000
rnn 300 3000
rnn 3000 500

conv2d 30 3 8000
conv2d 100 10 300
conv2d 100 50 40

lstm 300 1000
lstm 3000 500
lstm 30000 50
