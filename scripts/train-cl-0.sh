CUDA_VISIBLE_DEVICES=1 \
python3 -m train.CL.train \
     -modulo 5 \
     -seed 2 \
     -aux_data \
     -guesser_modulo 1 \
     -exp_name fair-b1s2t \
     -bin_name fair-b1s2t \
     -fair \
     > my_logs/fair-b1s2.txt 2>&1