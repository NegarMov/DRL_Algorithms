@echo off
pushd ..
python run.py ^
    --env_name "Ant-v4" ^
    --expert_data "_expert_data/expert_data_Ant-v4.pkl" ^
    --expert_policy_file "policy/experts/Ant.pkl" ^
    --save_params ^
    --n_iter 1 ^
    --no_gpu
popd