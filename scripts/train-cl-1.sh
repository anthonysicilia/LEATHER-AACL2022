CUDA_VISIBLE_DEVICES=0 \
python3 -m train.CL.train \
     -modulo 5 \
     -seed 3 \
     -exp_name b0s3t \
     -bin_name b0s3t \
     > my_logs/b0s3.txt 2>&1
