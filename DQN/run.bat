@echo off
python run.py ^
    --env_name "CartPole-v1" ^
    --episodes 1 ^
    --df 0.9 ^
    --e 1.0 ^
    --e_min 0.1 ^
    --e_decay_rate 0.99 ^
    --hidden_layers 2 ^
    --hidden_size 32 ^
    --lr 5e-3 ^
    --batch_size 128 ^
    --dueling ^
    --eval
