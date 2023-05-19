#!/usr/bin/env python3
# coding=utf-8
"""
Modified from https://github.com/rosinality/style-based-gan-pytorch/blob/master/projector.py
"""
import argparse
import os
import glob
import math

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from piq import LPIPS

from my_utils import create_folder

RESOLUTION = 256
use_w_space = True
repeat_w = False  # if False, opt w+ space
num_layers = 14  # 14 for stylegan w+ space with stylegan_celeba_partial256
use_z_plus_space = False  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
genforce_model = 'stylegan_celeba_partial256'


def load_one_image(img_file):
    img = Image.open(img_file)

    # first lower the resolution from real images to emulate GMI low-resolution GAN effects
    lower_resolution = RESOLUTION
    if lower_resolution != RESOLUTION:
        img = F.resize(img, (lower_resolution, lower_resolution))

    img = F.resize(img, (RESOLUTION, RESOLUTION))

    img = np.array(img, dtype=np.uint8)
    img = img.transpose(2, 0, 1)
    assert len(img.shape) == 3  # assumes color images and no alpha channel
    img = torch.from_numpy(img).float() / 255.
    # img = (img - 0.5) / 0.5  # normalize pixels into [-1, 1]

    return img.unsqueeze(0)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def get_generator(batch_size, device):
    global use_w_space
    from genforce import my_get_GD
    if not genforce_model.startswith('stylegan'):
        use_w_space = False
    use_discri = False
    print('use_w_space', use_w_space)
    generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w, use_z_plus_space=use_z_plus_space)
    return generator


@torch.no_grad()
def find_close_latent_vector(imgs, generator, w_candidates, special_noises, percept):
    img_gen = generator(w_candidates, special_noises=special_noises)  # pixel in [0, 1]
    img_gen = F.resize(img_gen, (RESOLUTION, RESOLUTION))

    # p_loss = percept(img_gen, imgs, normalize=True).squeeze()
    p_loss = percept(img_gen, imgs.expand(len(img_gen), -1, -1, -1)).squeeze()
    ind = torch.argmin(p_loss)
    print(ind)
    return w_candidates[ind]


def main(source=None, auto_aligned=False, find_close=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto_aligned', action='store_true', help='Indicate the image has been auto aligned.')
    parser.add_argument('--find_close', action='store_true', help='Find similar latent w as the starting point.')
    parser.add_argument('source', help='Image to be aligned. Multiple files separated by ,')

    if source is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args([source, ])
        args.auto_aligned = auto_aligned
        args.find_close = find_close

    print(args)

    device = 'cuda'
    latent_dim = 512
    lr = 0.1  # originally, I used 0.1, but according to image2stylegan++, I changed to 0.01
    percept_ce = 1.
    mse_ce = 0.0
    batch_size = 1
    step = 1000

    assert use_w_space and not repeat_w and not use_z_plus_space

    create_folder('./embedded_imgs')
    create_folder('./tmp')

    generator = get_generator(batch_size, device)

    percept = LPIPS(replace_pooling=True, reduction="none")

    batches = [
        [x] for x in args.source.split(',') if x and os.path.isfile(x)
    ]

    all_projected_filenames = []

    for batch in batches:
        imgs = [load_one_image(f) for f in batch]
        imgs = torch.cat(imgs, dim=0)

        assert len(imgs) == 1, 'currently embed one image at a time'

        imgs = imgs.to(device)

        org_imgs = imgs.clone()

        special_noises = []

        if args.auto_aligned:
            align_i = int(batch[0].split('_')[-1].split('.')[0])
            t_special_noises = torch.load('./random4x4_samples/random4x4_stylegan_stochastic_noise.pt')
            for noise_i in t_special_noises:
                if noise_i is not None:
                    noise_i = noise_i[align_i:1+align_i].detach().clone().repeat(1, 1, 1, 1).to(device)
                special_noises.append(noise_i)

        if args.find_close:
            n_mean_latent = 100
        else:
            n_mean_latent = 10000

        with torch.no_grad():
            latent_z_inputs = torch.randn(n_mean_latent, latent_dim, device=device)
            latent_w = generator.G.mapping(latent_z_inputs)['w']
            print(f'latent_z_inputs.shape: {latent_z_inputs.shape}')
            print(f'latent_w.shape: {latent_w.shape}')
            if args.find_close:
                latent_w_mean = find_close_latent_vector(imgs, generator, latent_w.unsqueeze(1).expand(-1, num_layers, -1), special_noises, percept)
                latent_w_mean = latent_w[0]
            else:
                latent_w_mean = latent_w.mean(0)
            print(f'latent_w_mean.shape: {latent_w_mean.shape}')

        # optimize the w space instead of z space
        latent_in = latent_w_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if not repeat_w:
            latent_in = latent_in.unsqueeze(1).repeat(1, num_layers, 1)

        with torch.no_grad():
            filename = f'./tmp/project_mean_{os.path.splitext(os.path.basename(batch[0]))[0]}'
            img_gen = generator(latent_w_mean.expand_as(latent_in), special_noises=special_noises)  # pixel in [0, 1]
            img_gen = F.resize(img_gen, (RESOLUTION, RESOLUTION))

            if imgs.shape[0] > 1:
                save_image(torch.stack([imgs, img_gen, ], dim=1).view(2*imgs.shape[0], *imgs.shape[1:]), f'{filename}.png', nrow=2)
            else:
                save_image(torch.cat([imgs, img_gen, ], dim=0), f'{filename}.png')

        latent_in.requires_grad_(True)
        optimizer = optim.Adam([latent_in, ], lr=lr)

        print('repeat_w', repeat_w)
        pbar = tqdm(range(step))

        for i in pbar:
            t = i / step
            lr_i = get_lr(t, lr)
            optimizer.param_groups[0]['lr'] = lr_i
            img_gen = generator(latent_in, special_noises=special_noises)  # pixel in [0, 1]
            img_gen = F.resize(img_gen, (RESOLUTION, RESOLUTION))

            # p_loss = percept(img_gen, imgs, normalize=True).sum()
            p_loss = percept(img_gen, imgs).mean()
            mse_loss = nn.functional.mse_loss(img_gen, imgs)

            loss = percept_ce * p_loss + mse_ce * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f'loss: {loss.item():.4f}'
                    f' perceptual: {p_loss.item():.4f} * {percept_ce};'
                    f' mse: {mse_loss.item():.4f} * {mse_ce}; lr: {lr_i:.4f}'
                )
            )

        filename = f'./embedded_imgs/{os.path.splitext(os.path.basename(batch[0]))[0]}'
        all_projected_filenames.append(filename)
        with torch.no_grad():
            t_latent_in = latent_in.detach()
            img_gen = generator(t_latent_in, special_noises=special_noises)
            img_gen = F.resize(img_gen, (RESOLUTION, RESOLUTION))
            torch.save(img_gen, f'{filename}_img.pt')
            torch.save(t_latent_in, f'{filename}_latent.pt')
            torch.save(special_noises, f'{filename}_noises.pt')
            if imgs.shape[0] > 1:
                save_image(torch.stack([imgs, img_gen, ], dim=1).view(2*imgs.shape[0], *imgs.shape[1:]), f'{filename}.png', nrow=2)
            else:
                save_image(torch.cat([imgs, img_gen, ], dim=0), f'{filename}.png')

    return all_projected_filenames


if __name__ == '__main__':
    main()
