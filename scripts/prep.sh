CUDA_VISIBLE_DEVICES=0 \
python3 -m train.SL.train \
     -modulo 7 \
     -exp_name prep \
     -bin_name prep \
     > my_logs/sl.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python3 -m train.Oracle.train \
     -img_feat res \
     -exp_name prep \
     -bin_name prep \
     > my_logs/oracle.txt 2>&1 &