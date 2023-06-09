#!/usr/bin/env python3
# coding=utf-8
import argparse
import glob
import os
import sys

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.utils import save_image

from facenet_pytorch import InceptionResnetV1
import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
import net_sphere
from my_utils import crop_img, resize_img, normalize
from my_target_models import get_model, get_input_resolution
from my_get_fr_models import get_fr_models, get_embedding_for_img_file, get_ensemble_embedding_for_img_file


def myprint(*args, **kwargs):
    if False:
        print(*args, **kwargs)


def normalize_tensor(inputs, arch_name, image_resolution, crop_face=False):
    if crop_face:
        inputs = resize_img(inputs, 256)
        inputs = inputs[..., 34:214, 40:220]
    else:
        inputs = crop_img(inputs, arch_name)
    inputs = resize_img(inputs, image_resolution)
    inputs = normalize(inputs*255., arch_name)
    myprint(f'loaded inputs shape: {inputs.shape}')
    return inputs


@torch.no_grad()
def test_final_result(arch_name=None, dirs=None, external_args=None):

    if arch_name is None or dirs is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('arch_name', help='network architecture')
        parser.add_argument('root_dir', help='the dir of the final images')
        parser.add_argument('--latent_space', choices=['w', 'z', 'z+', 'w+'], help='evaluate batch with another model')
        parser.add_argument('genforce_model', choices=['pggan_celebahq1024', 'stylegan_celeba_partial256', 'stylegan_ffhq256', 'stylegan2_ffhq1024', 'stylegan_cat256', 'stylegan_car512', ], help='genforce gan model')
        parser.add_argument('--bs', default=8, type=int, help='batch size')

        args = parser.parse_args()
        external_args = args

        arch_name = args.arch_name
        batch_size = args.bs
        dirs = [args.root_dir, ]
    else:
        assert arch_name is not None and dirs is not None
        if isinstance(dirs, str):
            dirs = [dirs, ]
        batch_size = external_args.bs

    assert external_args is not None

    device = 'cuda'

    if arch_name == 'sphere20a':
        sphere20_theta_net = getattr(net_sphere, 'sphere20a')(use_theta=True)
        sphere20_theta_net.load_state_dict(torch.load('./sphere20a_20171020.pth'))
        sphere20_theta_net.to(device)

    if arch_name.startswith('fr_'):
        net, image_resolution, fr_threshold = get_fr_models(arch_name, device)
    else:
        net = get_model(arch_name, device)  # we test the results on the original network
        image_resolution = get_input_resolution(arch_name)

    try:
        label_logits_dict = torch.load(os.path.join('./centroid_data', arch_name, 'test/centroid_logits.pt'))
    except FileNotFoundError:
        print('Note: centroid_logits.pt is not found')
        label_logits_dict = None

    use_w_space = 'w' in external_args.latent_space
    repeat_w = '+' not in external_args.latent_space   # if False, opt w+ space
    # num_layers = 14  # 14 for stylegan w+ space
    # num_layers = 18  # 14 for stylegan w+ space with stylegan_celebahq1024

    genforce_model = external_args.genforce_model
    if not genforce_model.startswith('stylegan'):
        use_w_space = False

    if external_args.latent_space == 'z+':
        use_z_plus_space = True  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
        use_w_space = True
    else:
        use_z_plus_space = False

    def get_generator(batch_size, device):
        from genforce import my_get_GD
        use_discri = False
        generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w, use_z_plus_space=use_z_plus_space)
        return generator

    generator = get_generator(batch_size, device)

    all_confs = []
    correct_cnt = 0
    topk_correct_cnt = 0
    total_cnt = 0
    l2_dist = []
    conf_diff_scores = []
    all_images = []
    for root_dir_i, root_dir in enumerate(dirs):
        if arch_name.startswith('fr_'):
            print('get_embedding_for_img_file for', external_args.targets[root_dir_i])
            all_targets = get_embedding_for_img_file(external_args.targets[root_dir_i], arch_name, net, image_resolution, device).repeat(batch_size, 1)
        else:
            tensor_files = sorted(glob.glob(os.path.join(root_dir, 'img_*.pt')))
            all_tensor_files = {}
            for f in tensor_files:
                f_basename = os.path.basename(f).split('_')
                id_in_batch = int(f_basename[2][2:])
                target = int(f_basename[1][5:])
                all_tensor_files[id_in_batch] = (target, f)

            all_targets = [all_tensor_files[k][0] for k in sorted(all_tensor_files.keys())]

        latent_in = torch.load(os.path.join(root_dir, 'latent_inputs.pt')).to(device)
        special_noises = torch.load(os.path.join(root_dir, 'special_noises.pt'), map_location=device)

        assert batch_size == len(all_targets)

        images = generator(latent_in, special_noises=special_noises)
        all_images.append(images)
        save_image(images, './tmp/my_test_confidence.png')
        images = normalize_tensor(images, arch_name, image_resolution)

        outputs = net(images.to(device))
        if arch_name.startswith('fr_'):
            logits = None
        elif arch_name == 'sphere20a':
            outputs = outputs[0]
            logits = sphere20_theta_net(images.to(device)).cpu()
        else:
            logits = outputs.detach().cpu()

        if arch_name.startswith('fr_'):
            print('outputs.shape', outputs.shape, 'all_targets.shape', all_targets.shape)
            simi = nn.functional.cosine_similarity(outputs, all_targets)
            all_confs.append(torch.clamp(simi, min=0.).tolist())
            _correct_cnt = simi>=fr_threshold
            # print(_correct_cnt)
            _correct_cnt = _correct_cnt.sum().item()
            topk_correct_cnt += _correct_cnt
            correct_cnt += _correct_cnt
            total_cnt += batch_size
            print('Verifier accuracy:', 100*_correct_cnt/batch_size)
            print('Confidence:', ','.join(map(lambda x: f'{x:.6f}', simi.tolist())))
        else:
            conf_diff_scores.extend(compute_confidence_score(outputs, all_targets))
            outputs = nn.functional.softmax(outputs, dim=1)
            conf_res = []
            for i, t in enumerate(all_targets):
                conf_res.append(f'{outputs[i][t].item():.4f}')
                if arch_name == 'sphere20a':
                    label_logits_dict and l2_dist.append(torch.dist(logits[i], label_logits_dict[t]).item())
                else:
                    label_logits_dict and l2_dist.append(torch.dist(logits[i], label_logits_dict[t]).item())
            all_confs.append([outputs[i][t].item() for i, t in enumerate(all_targets)])
            myprint('confidence:', '              '.join(conf_res))
            k = 5
            myprint(f'top-{k} labels')
            topk_conf, topk_class = torch.topk(outputs, k, dim=1)
            myprint(topk_conf)
            myprint(topk_class)
            total_cnt += len(all_targets)
            for i, t in enumerate(all_targets):
                if topk_class[i][0] == t:
                    correct_cnt += 1
                if t in topk_class[i]:
                    topk_correct_cnt += 1
    l2_dist = len(l2_dist) and sum(l2_dist)/len(l2_dist)
    myprint('l2 dist:', l2_dist)
    conf_diff_score = len(conf_diff_scores) and sum(conf_diff_scores)/len(conf_diff_scores)
    myprint(f'conf_diff_scores {len(conf_diff_scores)}: {conf_diff_score}')

    return all_confs, correct_cnt, topk_correct_cnt, total_cnt, l2_dist, conf_diff_score


def compute_confidence_score(outputs, all_targets):
    outputs = outputs.clone()
    target_conf_scores = []
    for i, t in enumerate(all_targets):
        output = outputs[i]
        conf_score = output[t].item()
        output[t] = 0.
        other_max_score = output.max().item()
        target_conf_scores.append(conf_score-other_max_score)
    return target_conf_scores


if __name__ == '__main__':
    test_final_result()
