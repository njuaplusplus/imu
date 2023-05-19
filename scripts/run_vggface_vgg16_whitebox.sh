#!/bin/zsh
for target in 1031
do
  python3 my_whitebox_imu.py --bs=8 --do_flip --exp_name="vgg16_imu_0.1_$target" --arch_name="vgg16" --lr 0.1 --loss_class_ce=1. --epochs=2000 --target=$target --embedded_filenames './my_examples/2std' --bound_latent_w_by_clip_ce=0.1
done



