@echo off
pushd ..
python run.py ^
    --env_name "Pendulum-v1" ^
    --episodes 1000 ^
    --warmup_episodes 800 ^
    --batch_size 128 ^
    --df 0.99 ^
    --noise_scale 0.1 ^
    --actor_lr 1e-4 ^
    --critic_lr 1e-3 ^
    --hidden_layers 2
    --hidden_size 256
popd