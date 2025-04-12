@echo off
python run.py ^
    --episodes 1 ^
    --df 0.9 ^
    --e 1.0 ^
    --network_sync_rate 10 ^
    --hidden_layers 1 ^
    --hidden_size 16 ^
    --lr 1e-3 ^
    --batch_size 32 ^
    --ddqn ^
    --eval
