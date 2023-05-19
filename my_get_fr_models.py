#!/usr/bin/env python3
# coding=utf-8
import glob
import os
import shutil

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.utils import save_image

import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152

from my_utils import resize_img, normalize


th_dict = {'fr_ir152': (0.094632, 0.166788, 0.227922),
           'fr_irse50': (0.144840, 0.241045, 0.312703),
           'fr_facenet': (0.256587, 0.409131, 0.591191),
           'fr_mobile_face': (0.183635, 0.301611, 0.380878)}


def get_fr_resolution(arch_name):
    resolution_dict = {'fr_ir152': (112, 112),
                       'fr_irse50': (112, 112),
                       'fr_mobile_face': (112, 112),
                       'fr_facenet': (160, 160)}
    return resolution_dict[arch_name]


def get_fr_models(arch_name, device):
    if arch_name == 'fr_ir152':
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
    if arch_name == 'fr_irse50':
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
    if arch_name == 'fr_mobile_face':
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
    if arch_name == 'fr_facenet':
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
    fr_model.to(device)
    fr_model.eval()

    resolution = get_fr_resolution(arch_name)

    th01, th001, th0001 = th_dict[arch_name]

    fr_threshold = th001

    return fr_model, resolution, fr_threshold


class EnsembleModel(nn.Module):
    def __init__(self, subject_arch_name, device, bs):
        super().__init__()
        self.arch_name_list = []
        self.model_list = []
        self.resolution_list = []
        for arch_name in th_dict.keys():
            if arch_name != subject_arch_name:
                fr_model, resolution, _ = get_fr_models(arch_name, device)
                self.arch_name_list.append(arch_name)
                self.model_list.append(fr_model)
                self.resolution_list.append(resolution)
        self.to(device)
        self.device = device
        self.bs = bs
        self.num_models = len(self.model_list)
        print('using ensemble models', ','.join(self.arch_name_list))

    def forward(self, inputs):
        """
            inputs is in [0, 1]
        """
        outputs = []
        for arch_name, fr_model, resolution in zip(self.arch_name_list, self.model_list, self.resolution_list):
            outputs.append(fr_model(normalize(resize_img(inputs*255., resolution), arch_name)))
        outputs = torch.cat(outputs, dim=0)
        return outputs


# def cos_simi(emb_before_pasted, emb_target_img):
#     return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
#                       / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))


def load_one_image(img_file, resolution):
    img = Image.open(img_file)

    if not isinstance(resolution, tuple):
        img = F.resize(img, (resolution, resolution))
    else:
        img = F.resize(img, resolution)

    img = np.array(img, dtype=np.uint8)
    img = img.transpose(2, 0, 1)
    assert len(img.shape) == 3  # assumes color images and no alpha channel
    img = torch.from_numpy(img).float() / 255.
    # img = (img - 0.5) / 0.5  # normalize pixels into [-1, 1]

    return img.unsqueeze(0)


@torch.no_grad()
def get_embedding_for_img_file(img_file, arch_name, model, resolution, device):
    img1 = load_one_image(img_file, resolution).to(device)
    img1 = normalize(img1*255, arch_name)
    emb1 = model(img1)
    # print(emb1.shape)
    return emb1


@torch.no_grad()
def get_ensemble_embedding_for_img_file(img_file, arch_name, model, bs):
    img = load_one_image(img_file, 256).to(model.device)
    emb = model(img)
    repeated_emb = []
    for i in range(len(emb)):
        repeated_emb.append(emb[i:i+1].repeat(bs, 1))
    emb = torch.cat(repeated_emb, dim=0)
    # print(emb.shape)
    # print(emb)
    # exit()
    return emb


if __name__ == '__main__':
    for arch_name in ['fr_ir152', 'fr_irse50', 'fr_mobile_face', 'fr_facenet', ]:
        get_fr_models(arch_name, 'cpu')
