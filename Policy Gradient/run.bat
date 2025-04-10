@echo off
python run.py ^
    --env_name "CartPole-v1" ^
    --episodes 2 ^
    --df 1.0 ^
    --hidden_layers 1 ^
    --hidden_size 16 ^
    --lr 1e-2 ^
    --eval
