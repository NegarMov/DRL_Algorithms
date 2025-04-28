@echo off
pushd ..
python run.py ^
    --env_name "PongNoFrameskip-v4" ^
    --episodes 5 ^
    --eval ^
    --checkpoint 700 ^
    --e 0.05
popd