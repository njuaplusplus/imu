#!/bin/zsh
arch="fr_irse50"
for target in './my_examples/real_chengwu_1.jpg' './my_examples/real_xunzhou_2.jpg'
do
  python3 my_whitebox_imu.py --bs=4 --do_flip --exp_name="${arch}_imu_$target" --arch_name="${arch}" --lr 0.1 --loss_class_ce=1. --epochs=2000 --target=$target --embedded_filenames './my_examples/me_bs4'
done
