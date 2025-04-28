@echo off
pushd ..
python run.py ^
    --env_name "HalfCheetah-v5" ^
    --episodes 900 ^
    --df 0.95 ^
    --hidden_layers 2 ^
    --hidden_size 64 ^
    --lr 1e-2
popd