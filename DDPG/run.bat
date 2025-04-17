@echo off
python run.py ^
    --env_name "Pendulum-v1" ^
    --episodes 1000 ^
    --df 0.9 ^
    --noise 0.1 ^
    --noise_min 0.01 ^
    --noise_decay_rate 0.995 ^
    --hidden_layers 2 ^
    --hidden_size 32 ^
    --lr 1e-3 ^
    --batch_size 128 ^
