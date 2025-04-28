@echo off
pushd ..
python run.py ^
    --episodes 15000 ^
    --is_slippery ^
    --e 1.0 ^
    --e_decay_rate 1e-4 ^
    --df 0.9 ^
    --lr 0.9
popd