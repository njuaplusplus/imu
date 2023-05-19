#!/usr/bin/env python3
# coding=utf-8
import torch
from torch import nn
import torchvision.models as models

import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
from facenet_pytorch import InceptionResnetV1
import net_sphere


def get_model(arch_name, device, use_dropout=False):
    if arch_name == 'vgg16bn':
        net = vgg_m_face_bn_dag.vgg_m_face_bn_dag('./CACHE/vgg_m_face_bn_dag.pth', use_dropout=use_dropout).to(device)
    elif arch_name == 'resnet50':
        net = resnet50_scratch_dag.resnet50_scratch_dag('./CACHE/resnet50_scratch_dag.pth').to(device)
    elif arch_name == 'vgg16':
        net = vgg_face_dag.vgg_face_dag('./CACHE/vgg_face_dag.pth', use_dropout=use_dropout).to(device)
    elif arch_name == 'inception_resnetv1_vggface2':
        net = InceptionResnetV1(classify=True, pretrained='vggface2').to(device)
    elif arch_name == 'inception_resnetv1_casia':
        net = InceptionResnetV1(classify=True, pretrained='casia-webface').to(device)
    elif arch_name == 'sphere20a':
        net = getattr(net_sphere, 'sphere20a')(use_dropout=use_dropout)
        net.load_state_dict(torch.load('./CACHE/sphere20a_20171020.pth'))
        net.to(device)
    else:
        raise AssertionError('wrong arch_name')
    net.eval()
    return net


def get_input_resolution(arch_name):
    resolution = 224
    # to_grayscale = False
    if arch_name.startswith('inception_resnetv1'):
        resolution = 160
    elif arch_name == 'sphere20a':
        resolution = (112, 96)
    elif arch_name.startswith('ccs19ami'):
        resolution = 64
        if 'rgb' not in arch_name:
            # to_grayscale = True
            pass
    elif arch_name in ['azure', 'clarifai', ]:
        resolution = 256
    elif arch_name == 'car_resnet34':
        resolution = 400

    return resolution
