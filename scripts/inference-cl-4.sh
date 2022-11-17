CUDA_VISIBLE_DEVICES=0 \
python3 -m train.GamePlay.inference \
    -exp_name ba2s2e64 \
    -load_bin_path bin/CL/ba2s2t2022_04_12_05_02/model_ensemble_addnTrain_ba2s2t_E_64 \
    > my_logs/inference/ba2s2e64.txt 2>&1

# CUDA_VISIBLE_DEVICES=0 \
# python3 -m train.GamePlay.inference \
#     -exp_name ba2s2e5 \
#     -load_bin_path bin/CL/ba2s2t2022_04_12_05_02/model_ensemble_addnTrain_ba2s2t_E_5 \
#     > my_logs/inference/ba2s2e5.txt 2>&1

# CUDA_VISIBLE_DEVICES=0 \
# python3 -m train.GamePlay.inference \
#     -exp_name ba2s2e25 \
#     -load_bin_path bin/CL/ba2s2t2022_04_12_05_02/model_ensemble_addnTrain_ba2s2t_E_25 \
#     > my_logs/inference/ba2s2e25.txt 2>&1

# CUDA_VISIBLE_DEVICES=0 \
# python3 -m train.GamePlay.inference \
#     -exp_name ba2s2e50 \
#     -load_bin_path bin/CL/ba2s2t2022_04_12_05_02/model_ensemble_addnTrain_ba2s2t_E_50 \
#     > my_logs/inference/ba2s2e50.txt 2>&1

# CUDA_VISIBLE_DEVICES=0 \
# python3 -m train.GamePlay.inference \
#     -exp_name ba2s2e75 \
#     -load_bin_path bin/CL/ba2s2t2022_04_12_05_02/model_ensemble_addnTrain_ba2s2t_E_75 \
#     > my_logs/inference/ba2s2e75.txt 2>&1

# CUDA_VISIBLE_DEVICES=0 \
# python3 -m train.GamePlay.inference \
#     -exp_name ba2s2e99\
#     -load_bin_path bin/CL/ba2s2t2022_04_12_05_02/model_ensemble_addnTrain_ba2s2t_E_99 \
#     > my_logs/inference/ba2s2e99.txt 2>&1