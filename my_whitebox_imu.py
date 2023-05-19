#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import torchvision.models as models

import net_sphere

from my_utils import create_folder, Tee
from my_whitebox_imu_helper import imu_attack
from my_target_models import get_model, get_input_resolution
from my_get_fr_models import get_fr_models, EnsembleModel, get_fr_resolution
from my_get_fr_models import get_embedding_for_img_file, get_ensemble_embedding_for_img_file


random.seed(0)


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.device = device

    args.final_img_dirname = f'final_images/{args.exp_name}'
    args.tmp_img_dirname = f'generations/{args.exp_name}'

    create_folder(args.tmp_img_dirname)
    Tee(os.path.join(args.tmp_img_dirname, 'output.log'), 'w')

    create_folder(f'{args.tmp_img_dirname}/images/')
    create_folder(args.final_img_dirname)

    verifier_device = torch.device(f'cuda:{torch.cuda.device_count()-1}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    if args.arch_name.startswith('fr_'):
        # the subject model is a verifier
        assert not args.use_cwloss, 'FR model only uses cosine loss'

        if args.ensemble:
            net = EnsembleModel(args.arch_name, device, args.bs)
            args.image_resolution = get_fr_resolution(args.arch_name)
        else:
            net, args.image_resolution, _ = get_fr_models(args.arch_name, device)
        net_verifier, _, args.fr_threshold_verifier = get_fr_models(args.arch_name, verifier_device)

        def cos_simi(emb_before_pasted, emb_target_img):
            return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                              / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))

        def cos_loss(outputs, targets):
            return 1 - cos_simi(outputs, targets)

        args.criterion = cos_loss

        assert os.path.isfile(args.target)

        if args.ensemble:
            args.targets = get_ensemble_embedding_for_img_file(args.target, args.arch_name, net, args.bs)
            args.verifier_targets = get_embedding_for_img_file(args.target, args.arch_name, net_verifier, args.image_resolution, verifier_device).repeat(args.bs, 1)
        else:
            args.targets = get_embedding_for_img_file(args.target, args.arch_name, net_verifier, args.image_resolution, verifier_device).repeat(args.bs, 1)

    else:
        # the subject model is a classifier
        net = get_model(args.arch_name, device, args.use_dropout)
        net_verifier = get_model(args.arch_name, verifier_device, args.use_dropout)
        args.image_resolution = get_input_resolution(args.arch_name)

        if args.arch_name == 'sphere20a':
            args.criterion = net_sphere.MyAngleLoss()
        else:
            args.criterion = nn.CrossEntropyLoss()

        if args.use_cwloss:
            true_target = int(args.target)  # only one target for the batch

            def my_cw_loss(outputs, targets):
                true_target_oh_ind = torch.zeros_like(outputs)
                assert len(outputs.shape) == 2 and len(outputs) == args.bs
                true_target_oh_ind[:, true_target] = 1.
                true_target_act_val = torch.sum(true_target_oh_ind*outputs, 1)
                inf = 1e6
                maxother_act_val = torch.max((1-true_target_oh_ind)*outputs-true_target_oh_ind*inf, 1)[0]
                true_target_cw_loss = (maxother_act_val - true_target_act_val).mean()

                return true_target_cw_loss

            args.criterion = my_cw_loss

        args.targets = list(map(int, args.target.split(',')))
        print(args.targets)

    print(args)

    args.embedded_filenames = args.embedded_filenames.split(',')
    print(args.embedded_filenames)

    imu_attack(args, net, net_verifier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--epochs', default=20000, type=int, help='optimization epochs')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--loss_class_ce', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--target', help='the only one target to invert, or multiple targets separated by ,')
    parser.add_argument('--trunc_psi', type=float, default=0.7, help='truncation percentage')
    parser.add_argument('--trunc_layers', type=int, default=8, help='num of layers to truncate')
    parser.add_argument('--all_ws_pt_file', default='./CACHE/stylegan_sample_z_stylegan_celeba_partial256_0.7_8_all_ws.pt', help='all ws pt file')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    parser.add_argument('--save_every', type=int, default=100, help='how often to save the intermediate results')
    parser.add_argument('--bound_latent_w_by_clip_ce', type=float, default=0.2, help='ce to bound w difference in each dimension.')
    parser.add_argument('--embedded_filenames', help='embedded filenames separete by ,')
    parser.add_argument('--use_cwloss', action='store_true', help='use cw loss')
    parser.add_argument('--ensemble', action='store_true', help='use ensemble model as the surrogate model to attack the subject verifier')

    args = parser.parse_args()
    # print(args)

    torch.backends.cudnn.benchmark = True

    run(args)


if __name__ == '__main__':
    main()
