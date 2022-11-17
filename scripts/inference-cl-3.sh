CUDA_VISIBLE_DEVICES=1 \
python3 -m train.GamePlay.inference \
    -exp_name b1s3e79 \
    -load_bin_path bin/CL/b1s3t2022_05_03_10_49/model_ensemble_addnTrain_b1s3t_E_79 \
    > my_logs/inference/b1s3e79.txt 2>&1

CUDA_VISIBLE_DEVICES=1 \
python3 -m train.GamePlay.inference \
    -exp_name b1s3e5 \
    -load_bin_path bin/CL/b1s3t2022_05_03_10_49/model_ensemble_addnTrain_b1s3t_E_5 \
    > my_logs/inference/b1s3e5.txt 2>&1

CUDA_VISIBLE_DEVICES=1 \
python3 -m train.GamePlay.inference \
    -exp_name b1s3e25 \
    -load_bin_path bin/CL/b1s3t2022_05_03_10_49/model_ensemble_addnTrain_b1s3t_E_25 \
    > my_logs/inference/b1s3e25.txt 2>&1

CUDA_VISIBLE_DEVICES=1 \
python3 -m train.GamePlay.inference \
    -exp_name b1s3e50 \
    -log_enchidden \
    -load_bin_path bin/CL/b1s3t2022_05_03_10_49/model_ensemble_addnTrain_b1s3t_E_50 \
    > my_logs/inference/b1s3e50.txt 2>&1

CUDA_VISIBLE_DEVICES=1 \
python3 -m train.GamePlay.inference \
    -exp_name b1s3e75 \
    -log_enchidden \
    -load_bin_path bin/CL/b1s3t2022_05_03_10_49/model_ensemble_addnTrain_b1s3t_E_75 \
    > my_logs/inference/b1s3e75.txt 2>&1

CUDA_VISIBLE_DEVICES=1 \
python3 -m train.GamePlay.inference \
    -exp_name b1s3e99\
    -log_enchidden \
    -load_bin_path bin/CL/b1s3t2022_05_03_10_49/model_ensemble_addnTrain_b1s3t_E_99 \
    > my_logs/inference/b1s3e99.txt 2>&1