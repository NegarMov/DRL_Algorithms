@echo off
pushd ..
python run.py ^
    --env_name "CartPole-v1" ^
    --episodes 300 ^
    --train_freq 8 ^
    --batch_size 128 ^
    --df 0.9999 ^
    --e 1.0 ^
    --e_min 0.1 ^
    --e_decay_rate 0.99 ^
    --lr 5e-3 ^
    --hidden_layers 2 ^
    --hidden_size 32 ^
    --dueling
popd