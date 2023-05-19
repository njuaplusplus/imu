#!/bin/zsh
arch="resnet50"
for target in 8536
do
  python3 my_whitebox_imu.py --bs=4 --do_flip --exp_name="${arch}_imu_cwloss_$target" --arch_name="${arch}" --lr 0.1 --loss_class_ce=1. --epochs=2000 --target=$target --embedded_filenames './my_examples/me_bs4' --use_cwloss
done
