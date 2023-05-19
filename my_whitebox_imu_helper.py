import collections
import glob
import math
import os
import random

from scipy.stats import truncnorm
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image
import torchvision.transforms.functional as F

from genforce import my_get_GD

from my_utils import clip, crop_img, resize_img, normalize, denormalize, clip_quantile_bound
from my_concat_final_images import concat_final_images


def adjust_lr(optimizer, initial_lr, epoch, epochs, rampdown=0.25, rampup=0.05):
    # from https://github.com/rosinality/style-based-gan-pytorch/blob/master/projector.py#L45
    t = epoch / epochs
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    lr = initial_lr * lr_ramp

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def fr_validate_one(inputs, targets, model, fr_threshold_verifier):
    emb = model(inputs)
    simi = nn.functional.cosine_similarity(emb, targets)
    cnt = (simi>=fr_threshold_verifier).sum().item()
    print('Verifier accuracy:', cnt*100/len(inputs))
    print('Confidence:', ','.join(map(lambda x: f'{x:.6f}', simi.tolist())))


def verify_accuracy(input, target, model, arch_name):
    def _split_into_even(data, n):
        assert len(data) % n == 0
        _s = len(data)//n
        _r = []
        for i in range(n):
            _ss = _s * i
            _r.append(data[_ss:_ss+_s])
        return _r

    def accuracy(output, target):
        batch_size = target.size(0)
        _, pred = output.max(dim=1)
        acc = ((pred == target).sum()*100./batch_size)
        return acc

    with torch.no_grad():
        device = next(model.parameters()).device
        n = 1
        inputs = _split_into_even(input, n)
        targets = _split_into_even(target, n)
        confidence_str = []
        acc = 0
        for input, target in zip(inputs, targets):
            if arch_name == 'sphere20a':
                output = model(input.to(device))[0]
            else:
                output = model(input.to(device))
            confidence = nn.functional.softmax(output, dim=1)
            assert n == 1, 'the following loop requires n == 1, or we need to recompute the i'
            for i, t in enumerate(target):
                confidence_str.append(f'{confidence[i][t]:.6f}')
            acc += accuracy(output.data, target.to(device)).item()

    print('Verifier accuracy:', acc/n)
    print('Confidence:', ','.join(confidence_str))


def imu_attack(args, target_model, verifier_model):

    best_cost = 1e4

    genforce_model = 'stylegan_celeba_partial256'
    latent_space = 'w+'

    use_z_plus_space = False
    use_w_space = 'w' in latent_space
    repeat_w = '+' not in latent_space
    if latent_space == 'z+':
        use_z_plus_space = True  # to use z+ space, set this and use_w_space to be true and repeat_w to be false
        use_w_space = True
    w_num_layers = 14  # 14 for img 256x256
    use_discri = False
    trunc_psi = args.trunc_psi
    trunc_layers = args.trunc_layers
    all_ws = None
    all_ws_pt_file = args.all_ws_pt_file

    # compute the statistical bound
    bound_latent_w_by_clip_ce = args.bound_latent_w_by_clip_ce
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)
    lrelu = nn.LeakyReLU(negative_slope=0.2)
    all_ws = torch.load(all_ws_pt_file).detach().to(args.device)
    print(f'all_ws.shape: {all_ws.shape}')
    all_ps = invert_lrelu(all_ws)
    all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
    direction_p_bound = bound_latent_w_by_clip_ce * all_p_stds
    direction_w_mins = lrelu(-direction_p_bound)
    direction_w_maxs = lrelu(direction_p_bound)
    print(f'direction_p_bound.shape: {direction_p_bound.shape}')

    generator, discri = my_get_GD.main(args.device, genforce_model, args.bs, args.bs, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w, use_z_plus_space=use_z_plus_space, trunc_psi=trunc_psi, trunc_layers=trunc_layers)
    if args.arch_name.startswith('fr_'):
        # this targets are the embedding
        targets = args.targets.to(args.device)
        args.nrow = args.bs
    else:
        assert isinstance(args.targets, list)

        args.nrow = int(args.bs / len(args.targets))

        # Make the same target adjacent
        dt = []
        for t in args.targets:
            for _ in range(int(args.bs / len(args.targets))):
                dt.append(t)
        args.targets = dt

        targets = torch.LongTensor(args.targets * (int(args.bs / len(args.targets)))).to(args.device)

    print(f'use_w_space = {use_w_space}\n'
          f'use_z_plus_space = {use_z_plus_space}\n'
          f'repeat_w = {repeat_w}\n'
          f'w_num_layers = {w_num_layers}')

    # load input w and special noises.
    with torch.no_grad():
        inputs = []
        special_noises = None
        for embedded in args.embedded_filenames:
            embedded_w = torch.load(f'{embedded}_latent.pt', map_location='cpu')
            inputs.append(embedded_w)
            if special_noises is None:
                special_noises = []
                for noise_i in torch.load(f'{embedded}_noises.pt', map_location='cpu'):
                    if noise_i is not None:
                        noise_i = [noise_i, ]
                    special_noises.append(noise_i)
            else:
                for _i, noise_i in enumerate(torch.load(f'{embedded}_noises.pt', map_location='cpu')):
                    if noise_i is not None:
                        special_noises[_i].append(noise_i)

    inputs = torch.cat(inputs, dim=0)
    # print('inputs.shape', inputs.shape)
    assert len(inputs) == args.bs

    for _i, noise_i in enumerate(special_noises):
        if noise_i is not None:
            special_noises[_i] = torch.cat(noise_i, dim=0).to(args.device)
        # print(_i, special_noises[_i].shape)

    with torch.no_grad():
        init_images = generator(inputs.to(args.device), special_noises=special_noises)
        save_image(init_images,
                   f'{args.tmp_img_dirname}/images/output_{0:05d}.png',
                   nrow=args.nrow)
        torch.save(init_images,
                   f'{args.tmp_img_dirname}/images/output_{0:05d}.pt')
        torch.save(inputs,
                   f'{args.tmp_img_dirname}/images/latent_input_{0:05d}.pt')
        torch.save(special_noises,
                   f'{args.tmp_img_dirname}/images/special_noises.pt')

    inputs = inputs.to(args.device)
    origin_inputs = inputs.detach().clone()

    direction = torch.zeros(1, *inputs.shape[1:], device=inputs.device, requires_grad=True)
    inputs = inputs + direction

    optimizer = optim.Adam([direction, ], lr=args.lr, betas=[0.9, 0.999], eps=1e-8)

    for epoch in range(1, args.epochs+1):
        # learning rate scheduling
        _lr = adjust_lr(optimizer, args.lr, epoch, args.epochs)

        # perform downsampling if needed
        fake = generator(inputs.to(args.device), special_noises=special_noises)
        if args.ensemble:
            input_images = fake  # ensemble fr models take in images in [0, 1]
        else:
            fake = crop_img(fake, args.arch_name)
            input_images = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)

        # horizontal flipping
        flip = random.random() > 0.5
        if args.do_flip and flip:
            input_images = torch.flip(input_images, dims=(3,))

        # forward pass
        optimizer.zero_grad()
        if use_discri:
            discri.zero_grad()
        generator.zero_grad()

        outputs = target_model(input_images.to(args.device))
        loss_class = args.criterion(outputs, targets.to(args.device))
        loss = args.loss_class_ce * loss_class

        if epoch % args.save_every==0:
            print(f'------------ epoch {epoch}----------')
            print('lr', _lr)
            print('total loss', loss.item())
            print(f'class loss (multiplied by {args.loss_class_ce})', loss_class.item())

            with torch.no_grad():
                fake = generator(inputs.detach().to(args.device), special_noises=special_noises)
                fake = crop_img(fake, args.arch_name)
                fake = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)
                if args.arch_name.startswith('fr_'):
                    if args.ensemble:
                        fr_validate_one(fake, args.verifier_targets, verifier_model, args.fr_threshold_verifier)
                    else:
                        fr_validate_one(fake, targets, verifier_model, args.fr_threshold_verifier)
                else:
                    verify_accuracy(fake, targets, verifier_model, args.arch_name)

        loss.backward()

        optimizer.step()

        # clip by statistical bound
        direction.data = clip_quantile_bound(direction.data, direction_w_mins, direction_w_maxs)
        inputs = origin_inputs + direction

        if best_cost > loss.item() or epoch == 0:
            best_inputs = inputs.data.clone()
            best_epoch = epoch
            best_cost = loss.item()

        if epoch % args.save_every==0 and (args.save_every > 0):
            with torch.no_grad():
                fake = generator(inputs.detach().to(args.device), special_noises=special_noises)
                fake = crop_img(fake, args.arch_name)
                fake = normalize(resize_img(fake*255., args.image_resolution), args.arch_name)
            torch.save(inputs,
                       f'{args.tmp_img_dirname}/images/latent_input_{epoch//args.save_every:05d}.pt')
            save_image(denormalize(fake, args.arch_name),
                       f'{args.tmp_img_dirname}/images/output_{epoch//args.save_every:05d}.png',
                       nrow=args.nrow)

    with torch.no_grad():
        latent_inputs = best_inputs.detach().clone()
        fake = generator(best_inputs.detach().to(args.device), special_noises=special_noises)
        # don't resize and downsample the images, but save the high-resolution images
        fake = normalize(fake*255., args.arch_name)

    for i in range(fake.shape[0]):
        if args.arch_name.startswith('fr_'):
            target = 0
        else:
            target = targets[i].item()
        save_filename = f'{args.final_img_dirname}/img_label{target:05d}_id{i:03d}_iter{best_epoch}.jpg'

        torch.save(latent_inputs[i], save_filename[:-4]+'.pt')

        image_np = denormalize(fake[i], args.arch_name).data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(save_filename)

    torch.save(latent_inputs, f'{args.final_img_dirname}/latent_inputs.pt')
    torch.save(special_noises, f'{args.final_img_dirname}/special_noises.pt')

    # concatenate all best images
    concat_final_images(args.final_img_dirname.rstrip('/'), nrow=args.nrow)
