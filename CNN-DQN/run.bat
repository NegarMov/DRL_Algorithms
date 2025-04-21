@echo off
python run.py ^
    --env_name "BreakoutDeterministic-v4" ^
    --episodes 300000 ^
    --log_freq 100 ^
    --warmup_steps 50000 ^
    --max_buffer_size 100000 ^
    --network_sync_rate 10000 ^
    --df 0.99 ^
    --e 1.0 ^
    --e_min 0.1 ^
    --e_decay_rate 1000000 ^
    --lr 2.5e-4 ^
    --batch_size 32 ^
    --ddqn
