@echo off
pushd ..
python run.py ^
    --env_name "HalfCheetah-v5" ^
    --episodes 5 ^
    --actor_hidden_layers 2 ^
    --actor_hidden_size 64 ^
    --critic_hidden_layers 2 ^
    --critic_hidden_size 64 ^
    --eval
popd