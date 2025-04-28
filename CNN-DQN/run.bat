@echo off
python run.py ^
    --env_name "PongNoFrameskip-v4" ^
    --episodes 10 ^
    --checkpoint 100 ^
    --max_buffer_size 50000 ^
    --warmup_steps 40000 ^
    --df 0.97 ^
    --e 1.0 ^
    --e_min 0.05 ^
    --e_decay_rate 0.99 ^
    --lr 2.5e-4 ^
    --batch_size 64 ^
    --eval ^
    --eval_checkpoint 700
