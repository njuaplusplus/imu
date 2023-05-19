#!/bin/zsh
dir_prefix="fr_irse50_imu"
for target in './my_examples/real_chengwu_1.jpg' './my_examples/real_xunzhou_2.jpg'
do
  for arch in 'fr_irse50' 'fr_ir152' 'fr_mobile_face' 'fr_facenet'
    do
      python3 my_concat_all_final_images.py --bs 4 --add_conf $arch $dir_prefix $target
    done
done
