#!/bin/bash
ray stop && ray start --head --dashboard-host=0.0.0.0 --dashboard-port 80
(sleep 5 && RAY_DEDUP_LOGS=0 python run_ray_cache.py --name cache:main --num-cpus 2 --num-gpus 0) &
(sleep 5 && RAY_DEDUP_LOGS=0 python run_ray_network_worker.py --name network:main --num-workers 5 --num-cpus 0.1 --num-gpus 0) &
(sleep 5 && RAY_DEDUP_LOGS=0 python run_ray_gpu_worker.py --name gpu --num-workers 3 --num-cpus 0.5 --num-gpus 0.2) &
(sleep 5 && RAY_DEDUP_LOGS=0 python run_ray_cpu_worker.py --name cpu:main --num-workers 6 --num-cpus 0.5 --num-gpus 0) &
sleep 5 && python3 run_runpod_worker.py

# # Name of the tmux session
# SESSION_NAME="ray_session"
#
# # Commands to run in each tmux tab
# commands=(
#     "ray stop && ray start --head --dashboard-host=0.0.0.0 --dashboard-port 80"
#     "sleep 10 && RAY_DEDUP_LOGS=0 python run_ray_cache.py --name cache:main --num-cpus 2 --num-gpus 0"
#     "sleep 10 && RAY_DEDUP_LOGS=0 python run_ray_network_worker.py --name network:main --num-workers 10 --num-cpus 0.1 --num-gpus 0"
#     "sleep 10 && RAY_DEDUP_LOGS=0 python run_ray_gpu_worker.py --name gpu --num-workers 3 --num-cpus 0.5 --num-gpus 0.2"
#     "sleep 10 && RAY_DEDUP_LOGS=0 python run_ray_cpu_worker.py --name cpu:main --num-workers 10 --num-cpus 0.5 --num-gpus 0"
#     "sleep 10 && python3 run_dispatcher.py"
#     "htop"
# )
#
# # Kill the session if it already exists to avoid duplication
# tmux kill-session -t $SESSION_NAME 2>/dev/null
#
# # Create a new tmux session but don't attach yet
# tmux new-session -d -s $SESSION_NAME -n "Main"
#
# # Start each command in its own tmux window
# for i in "${!commands[@]}"; do
#     if [ $i -eq 0 ]; then
#         # Start the first command in the initial window
#         tmux send-keys -t $SESSION_NAME "${commands[$i]}" C-m
#     else
#         # Create a new window and run the command
#         tmux new-window -t $SESSION_NAME -n "Tab$i"
#         tmux send-keys -t $SESSION_NAME:$(($i)) "${commands[$i]}" C-m
#     fi
# done
#
# # Attach to the tmux session
# tmux attach -t $SESSION_NAME
