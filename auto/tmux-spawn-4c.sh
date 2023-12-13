#!/bin/sh
CUDA_DEV=$1
CONDA_ENV=$2
SERVER=$3

if [ $# -lt 3 ]
then
    echo "Not enough Args Supplied. Aborting..."
    exit
fi

tmux new-session -d

tmux split-window -v
tmux split-window -h

tmux select-pane -t 0
tmux split-window -h

for n in {0..3};
do
    tmux select-pane -t $n
    tmux send-keys "export CUDA_VISIBLE_DEVICES=$CUDA_DEV" C-m
    tmux send-keys "conda activate $CONDA_ENV" C-m
    tmux send-keys 'cd ../client' C-m
    tmux send-keys "python main.py -s $SERVER" C-m
done

tmux -2 attach-session