@echo off
pushd ..
python run.py ^
    --env_name "HalfCheetah-v5" ^
    --episodes 5 ^
    --hidden_layers 2 ^
    --hidden_size 64 ^
    --eval
popd