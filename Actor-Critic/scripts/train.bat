@echo off
pushd ..
python run.py ^
    --env_name "HalfCheetah-v5" ^
    --episodes 900 ^
    --df 0.95 ^
    --lambda 0.95 ^
    --actor_hidden_layers 2 ^
    --actor_hidden_size 64 ^
    --critic_hidden_layers 2 ^
    --critic_hidden_size 64
popd