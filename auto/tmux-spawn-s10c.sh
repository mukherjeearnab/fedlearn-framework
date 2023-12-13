#!/bin/sh
CUDA_DEV=$1
CONDA_ENV=$2
SERVER=$3

tmux new-session -d

tmux split-window -h

tmux select-pane -t 0
tmux split-window -v

tmux select-pane -t 2
tmux split-window -v
tmux split-window -v
tmux split-window -v

tmux select-pane -t 2
tmux split-window -v
tmux split-window -v

tmux select-pane -t 2
tmux split-window -v

tmux select-pane -t 6
tmux split-window -v

tmux select-pane -t 1
tmux split-window -v
tmux split-window -v

tmux select-pane -t 1
tmux split-window -v

for n in {0..12};
do
    tmux select-pane -t $n
    tmux send-keys "export CUDA_VISIBLE_DEVICES=$CUDA_DEV" C-m
    tmux send-keys "conda activate $CONDA_ENV" C-m
done

tmux select-pane -t 0
tmux send-keys 'cd ../server' C-m
tmux send-keys "python main.py -p $SERVER" C-m

tmux select-pane -t 1
tmux send-keys 'cd ../kvstore' C-m
tmux send-keys "python main.py" C-m

tmux select-pane -t 2
tmux send-keys 'cd ../perflogger' C-m
tmux send-keys "python main.py" C-m

for n in {3..12};
do
    tmux select-pane -t $n
    tmux send-keys 'cd ../client' C-m
    tmux send-keys "python main.py -s $SERVER" C-m
done

tmux select-pane -t 0

tmux -2 attach-session