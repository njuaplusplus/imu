#!/bin/zsh
for target in 51 8536
do
  python3 my_blackbox_imu.py --arch_name="resnet50" --target=$target --population_size=1000 --exp_name="gpu_vggface2_resnet50_$target" --mutation_ce=1.0 --mutation_prob=0.1 --bound_latent_w_by_clip_ce=1.2 --num_poses=4 --embedded_filenames './my_examples/me_bs4'
done
