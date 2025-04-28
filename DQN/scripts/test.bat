@echo off
pushd ..
python run.py ^
    --env_name "CartPole-v1" ^
    --episodes 5 ^
    --hidden_layers 2 ^
    --hidden_size 32 ^
    --dueling ^
    --eval
popd