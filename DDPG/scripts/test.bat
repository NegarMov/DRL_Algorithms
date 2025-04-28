@echo off
pushd ..
python run.py ^
    --env_name "Pendulum-v1" ^
    --episodes 5 ^
    --hidden_layers 2 ^
    --hidden_size 256 ^
    --eval
popd