#!/bin/zsh
for target in 51 8536
do
  python3 my_blackbox_imu.py --arch_name="resnet50" --target=$target --test_only --result_dir="gpu_vggface2_resnet50" --num_poses=4 --embedded_filenames './my_examples/me_bs4'
done
