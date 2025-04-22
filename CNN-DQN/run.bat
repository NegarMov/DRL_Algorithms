@echo off
python run.py ^
    --env_name "PongNoFrameskip-v4" ^
    --episodes 1200 ^
    --log_freq 25 ^
    --warmup_steps 0 ^
    --max_buffer_size 200000 ^
    --network_sync_rate 10000 ^
    --df 0.99 ^
    --e 1.0 ^
    --e_min 0.1 ^
    --e_decay_rate 1.1e-6 ^
    --lr 1e-5 ^
    --batch_size 32 ^
    --dueling
