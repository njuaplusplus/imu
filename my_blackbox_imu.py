#!/usr/bin/env python3
# coding=utf-8
import argparse
import glob
import os
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import numpy as np

import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
from facenet_pytorch import InceptionResnetV1
import net_sphere
from my_utils import normalize, clip_quantile_bound, create_folder, Tee, add_conf_to_tensors, crop_and_resize
from my_target_models import get_model, get_input_resolution


random.seed(0)


class Sample:
    def __init__(self, value, fitness_score=-1):
        """
        value is a tensor
        """
        self.value = value
        self.fitness_score = fitness_score


class VectorizedPopulation:
    def __init__(self, population, fitness_scores, mutation_prob, mutation_ce, apply_noise_func, clip_func, compute_fitness_func):
        """
        population is a tensor with size N,512
        fitness_scores is a tensor with size N
        """
        self.population = population
        self.fitness_scores = fitness_scores
        self.mutation_prob = mutation_prob
        self.mutation_ce = mutation_ce
        self.apply_noise_func = apply_noise_func
        self.clip_func = clip_func
        self.compute_fitness_func = compute_fitness_func

        if self.fitness_scores is None:
            self.compute_fitness()

    def compute_fitness(self):
        bs = 10
        scores = []
        for i in range(0, len(self.population), bs):
            data = self.population[i:i+bs]
            scores.append(self.compute_fitness_func(data))
        self.fitness_scores = torch.cat(scores, dim=0)
        assert self.fitness_scores.ndim == 1 and self.fitness_scores.shape[0] == len(self.population)

    def find_elite(self):
        self.fitness_scores, indices = torch.sort(self.fitness_scores, dim=0, descending=True)
        self.population = self.population[indices]
        return Sample(self.population[0].detach().clone(), self.fitness_scores[0].item())

    def __get_parents(self, k):
        weights = F.softmax(self.fitness_scores, dim=0).tolist()
        parents_ind = random.choices(list(range(len(weights))), weights=weights, k=2*k)
        parents1_ind = parents_ind[:k]
        parents2_ind = parents_ind[k:]

        return parents1_ind, parents2_ind

    def __crossover(self, parents1_ind, parents2_ind):
        parents1_fitness_scores = self.fitness_scores[parents1_ind]
        parents2_fitness_scores = self.fitness_scores[parents2_ind]
        p = (parents1_fitness_scores / (parents1_fitness_scores + parents2_fitness_scores)).unsqueeze(1)  # size: N, 1
        parents1 = self.population[parents1_ind].detach().clone()  # size: N, 512
        parents2 = self.population[parents2_ind].detach().clone()  # size: N, 512
        if parents1.ndim == 3:  # direction in p+ space
            p = p.unsqueeze(1)
        mask = torch.rand_like(parents1)
        mask = (mask < p).float()
        return mask*parents1 + (1.-mask)*parents2

    def __mutate(self, children):
        mask = torch.rand_like(children)
        mask = (mask < self.mutation_prob).float()
        children = self.apply_noise_func(children, mask, self.mutation_ce)
        return self.clip_func(children)

    def produce_next_generation(self, elite):
        parents1_ind, parents2_ind = self.__get_parents(len(self.population)-1)
        children = self.__crossover(parents1_ind, parents2_ind)
        mutated_children = self.__mutate(children)
        self.population = torch.cat((elite.value.unsqueeze(0), mutated_children), dim=0)
        self.compute_fitness()

    def visualize_imgs(self, filename, generate_images_func, k=8):
        d = self.population[:k]
        out = generate_images_func(d, raw_img=True)
        vutils.save_image(out, filename)


def init_population(args):
    """
    find args.n images with highest confidence
    """
    # compute bound in p space
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)

    all_ws = torch.load(args.all_ws_pt_file).to(args.device)
    print(f'all_ws.shape: {all_ws.shape}')
    all_ps = invert_lrelu(all_ws)
    all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)

    # NOTE: perturabtion is in p+ space, direction should be transformed to w space before being applied
    bound_latent_w_by_clip_ce = args.bound_latent_w_by_clip_ce
    direction_p_bound = bound_latent_w_by_clip_ce * all_p_stds
    # direction_w_mins = lrelu(-direction_p_bound)
    # direction_w_maxs = lrelu(direction_p_bound)
    print('bound_latent_w_by_clip_ce', bound_latent_w_by_clip_ce)
    print(f'direction_p_bound.shape: {direction_p_bound.shape}')

    def clip_func(d):
        if d.ndim != 3:
            print(f'd.shape: {d.shape}. currently we use w+ space, whose shape is N x 14 x 512')
            raise AssertionError()
        return clip_quantile_bound(d, -direction_p_bound, direction_p_bound)

    def apply_noise_func(d, mask, ce):
        if d.ndim != 3:
            print(f'd.shape: {d.shape}. currently we use w+ space, whose shape is N x 14 x 512')
            raise AssertionError()
        p = d
        if args.randn_noise:
            noise = direction_p_bound * torch.randn_like(mask)
        else:
            noise = (2*direction_p_bound) * torch.rand_like(mask) - direction_p_bound
        noise = ce * noise
        p = p + mask*noise
        d = p
        return d

    if args.randn_noise:
        population = direction_p_bound * torch.randn(args.population_size, 14, 512, device=direction_p_bound.device)
    else:
        population = (2*direction_p_bound) * torch.rand(args.population_size, 14, 512, device=direction_p_bound.device) - direction_p_bound
    population = clip_func(population)

    print(f'init population shape: {population.shape}')
    fitness_scores = None

    return VectorizedPopulation(population, fitness_scores, args.mutation_prob, args.mutation_ce, apply_noise_func, clip_func, args.compute_fitness_func)


def genetic_algorithm(args, generator, generate_images_func):
    population = init_population(args)
    generations = args.generations
    for gen in range(generations):
        elite = population.find_elite()
        print(f'elite at {gen}-th generation: {math.exp(elite.fitness_score)}')
        population.visualize_imgs(os.path.join(args.exp_name, f'{gen}.png'), generate_images_func)

        if elite.fitness_score >= math.log(args.min_score):  # fitness_score is log_softmax
            return elite
        population.produce_next_generation(elite)

    # return the elite
    elite = population.find_elite()
    population.visualize_imgs(os.path.join(args.exp_name, f'{gen+1}.png'), generate_images_func)
    return elite


def compute_conf(net, arch_name, resolution, targets, imgs):
    if arch_name == 'sphere20a':
        sphere20_theta_net = getattr(net_sphere, 'sphere20a')(use_theta=True)
        sphere20_theta_net.load_state_dict(torch.load('./sphere20a_20171020.pth'))
        sphere20_theta_net.to('cuda')

    try:
        label_logits_dict = torch.load(os.path.join('./centroid_data', arch_name, 'test/centroid_logits.pt'))
    except FileNotFoundError:
        print('Note: centroid_logits.pt is not found')
        label_logits_dict = None

    outputs = net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name))
    if arch_name == 'sphere20a':
        outputs = outputs[0]
        # net.feature = True
        # logits = net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name)).cpu()
        # net.feature = False
        logits = sphere20_theta_net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name)).cpu()
    else:
        logits = outputs.cpu()
    logits_softmax = F.softmax(outputs, dim=1)

    target_conf = []

    k = 5
    print(f'top-{k} labels')
    topk_conf, topk_class = torch.topk(outputs, k, dim=1)
    correct_cnt = 0
    topk_correct_cnt = 0
    total_cnt = len(targets)
    l2_dist = []
    for i in range(len(targets)):
        t = targets[i]
        target_conf.append(logits_softmax[i, t].item())
        if label_logits_dict is not None:
            l2_dist.append(torch.dist(logits[i], label_logits_dict[t]).item())
        if topk_class[i][0] == t:
            correct_cnt += 1
        if t in topk_class[i]:
            topk_correct_cnt += 1
    # print('target conf:', target_conf)
    l2_dist = len(l2_dist) and sum(l2_dist)/len(l2_dist)
    print(arch_name)
    print(f'top1 acc: {correct_cnt}/{total_cnt} = {correct_cnt/total_cnt:.4f}')
    print(f'topk acc: {topk_correct_cnt}/{total_cnt} = {topk_correct_cnt/total_cnt:.4f}')
    print(f'l2_dist: {l2_dist:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--population_size', default=100, type=int, help='population size')
    parser.add_argument('--arch_name', default='ccs19ami_facescrub_rgb', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--log-interval', type=int, default=10, metavar='')
    parser.add_argument('--mutation_prob', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--mutation_ce', type=float, default=0.1, help='mutation coefficient')
    parser.add_argument('--generations', default=100, type=int, help='total generations')
    parser.add_argument('--target', default=0, type=int, help='the target label')
    parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std')
    parser.add_argument('--min_score', type=float, default=0.95, help='once reaching the score, terminate the attack')
    parser.add_argument('--all_ws_pt_file', default='./CACHE/stylegan_sample_z_stylegan_celeba_partial256_0.7_8_all_ws.pt', help='all ws pt file')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--randn_noise', action='store_true')
    parser.add_argument('--num_poses', default=4, type=int, help='the number of poses in a batch')
    parser.add_argument('--bound_latent_w_by_clip_ce', type=float, default=0.5, help='set the bound for p_space_bound mean+-x*std')
    parser.add_argument('--embedded_filenames', help='embedded filenames separete by ,')
    parser.add_argument('--result_dir', type=str, help='result directory to overwrite the default value.')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    args.exp_name = os.path.join('genetic_attack', args.exp_name)
    create_folder(args.exp_name)
    Tee(os.path.join(args.exp_name, 'output.log'), 'w')
    print(args)
    print(datetime.now())

    torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.device = device
    print(f'using device: {device}')

    net = get_model(args.arch_name, device)
    if args.arch_name == 'vgg16bn':
        args.test_arch_name = 'vgg16'
        result_dir = 'gpu_vggface_vgg16bn'
    elif args.arch_name == 'resnet50':
        args.test_arch_name = 'inception_resnetv1_vggface2'
        result_dir = 'gpu_vggface2_resnet50'
    elif args.arch_name == 'vgg16':
        args.test_arch_name = 'vgg16bn'
        result_dir = 'gpu_vggface_vgg16'
    elif args.arch_name == 'inception_resnetv1_vggface2':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_inceptionrnv1'
    elif args.arch_name == 'inception_resnetv1_casia':
        args.test_arch_name = 'sphere20a'
        result_dir = 'casia_inceptionrnv1'
    elif args.arch_name == 'sphere20a':
        args.test_arch_name = 'inception_resnetv1_casia'
        result_dir = 'gpu_vggface_sphere20a'
    else:
        raise AssertionError('wrong arch_name')

    if args.result_dir:
        result_dir = args.result_dir

    args.resolution = get_input_resolution(args.arch_name)
    args.test_resolution = get_input_resolution(args.test_arch_name)

    use_w_space = True
    repeat_w = True  # if False, opt w+ space
    # num_layers = 14  # 14 for stylegan w+ space
    # num_layers = 18  # 14 for stylegan w+ space with stylegan_celebahq1024

    genforce_model = 'stylegan_celeba_partial256'
    if not genforce_model.startswith('stylegan'):
        use_w_space = False

    def get_generator(batch_size, device):
        from genforce import my_get_GD
        use_discri = False
        generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w)
        return generator

    generator = get_generator(args.bs, device)

    args.embedded_filenames = args.embedded_filenames.split(',')
    print(args.embedded_filenames)

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

    inputs = torch.cat(inputs, dim=0).to(device)
    # print('inputs.shape', inputs.shape)
    assert len(inputs) == args.num_poses

    for _i, noise_i in enumerate(special_noises):
        if noise_i is not None:
            special_noises[_i] = torch.cat(noise_i, dim=0).to(device)
        # print(_i, special_noises[_i].shape)

    lrelu = nn.LeakyReLU(negative_slope=0.2)

    def generate_images_func(d, raw_img=False):
        if d.ndim != 3:
            print(f'd.shape: {d.shape}. currently we use w+ space, whose shape is N x 14 x 512')
            raise AssertionError()

        w = inputs.unsqueeze(0) + lrelu(d.unsqueeze(1))
        w = w.view(-1, 14, 512)
        # print('generate_images_func.w.shape', w.shape)
        _special_noises = []
        for noise_i in special_noises:
            if noise_i is not None:
                noise_i = noise_i.repeat([d.shape[0], ] + [1, ] * (noise_i.ndim-1))
            _special_noises.append(noise_i)
        if raw_img:
            return generator(w.to(device), special_noises=_special_noises)
        img = crop_and_resize(generator(w.to(device), special_noises=_special_noises), args.arch_name, args.resolution)
        return img

    def compute_fitness_func(w, img=None):
        if img is None:
            img = generate_images_func(w)
        else:
            img = crop_and_resize(img, args.arch_name, args.resolution)
        assert img.ndim == 4
        if args.arch_name == 'sphere20a':
            if args.test_only:
                pred = F.softmax(net(normalize(img*255., args.arch_name))[0], dim=1)
                print('use softmax')
                score = pred[:, args.target]
                return score
            else:
                pred = F.log_softmax(net(normalize(img*255., args.arch_name))[0], dim=1)
                score = pred[:, args.target]
                score = score.view(len(w), args.num_poses)
                score = score.mean(dim=1)
        else:
            if args.test_only:
                pred = F.softmax(net(normalize(img*255., args.arch_name)), dim=1)
                print('use softmax')
                score = pred[:, args.target]
                return score
            else:
                pred = F.log_softmax(net(normalize(img*255., args.arch_name)), dim=1)
                score = pred[:, args.target]
                score = score.view(len(w), args.num_poses)
                score = score.mean(dim=1)
        return score
    args.compute_fitness_func = compute_fitness_func

    if args.test_only:
        final_sample = torch.load(os.path.join('./genetic_attack', f'{result_dir}_{args.target}', 'final_w.pt'))
        d = final_sample.value.to(device).unsqueeze(0)
        score = math.exp(final_sample.fitness_score)
        print('recorded score:', score)
        imgs = generate_images_func(d, raw_img=True)

        targets = [args.target, ] * len(imgs)  # repeat targets to match the number of the images

        compute_conf(net, args.arch_name, args.resolution, targets, imgs)
        if args.test_arch_name != args.arch_name:
            compute_conf(get_model(args.test_arch_name, device), args.test_arch_name, args.test_resolution, targets, imgs)

        all_confs = compute_fitness_func(d, img=imgs)

        imgs = add_conf_to_tensors(imgs, all_confs)
        create_folder('./tmp')
        vutils.save_image(imgs, f'./tmp/all_{args.arch_name}_{args.target}_ge_images.png', nrow=args.num_poses)
        return

    res = genetic_algorithm(args, generator, generate_images_func)
    score = math.exp(res.fitness_score)
    print(f'final confidence: {score}')
    torch.save(res, os.path.join(args.exp_name, 'final_w.pt'))
    print(datetime.now())


if __name__ == '__main__':
    with torch.no_grad():
        main()
