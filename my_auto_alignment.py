#!/usr/bin/env python3
# coding=utf-8
import argparse
import glob
import os

from PIL import Image
import cv2
import scipy
import scipy.ndimage
import dlib

import numpy as np
import torch
import torchvision.transforms.functional as F
# from torchvision.utils import draw_keypoints
from torchvision.utils import save_image

from my_utils import create_folder


def get_landmark(filepath, predictor, img=None):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    if img is None:
        img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    shape = None
    for k, d in enumerate(dets):
        shape = predictor(img, d)

    if shape is None:
        print(f'{filepath} has no detected faces!')

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def visualize_landmark(img, lm, return_ratio=False):
    lm_chin = lm[0: 17]  # left-right
    # lm_eyebrow_left = lm[17: 22]  # left-right
    # lm_eyebrow_right = lm[22: 27]  # left-right
    # lm_nose = lm[27: 31]  # top-down
    # lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    # lm_mouth_inner = lm[60: 68]  # left-clockwise

    img = img.copy()
    pts = lm_chin
    for i in range(1, len(pts)):
        ptA = tuple(pts[i - 1])
        ptB = tuple(pts[i])
        cv2.line(img, ptA, ptB, (240, 10, 157), 2)
    for t_lm in (lm_eye_left, lm_eye_right, lm_mouth_outer):
        hull = cv2.convexHull(t_lm)
        cv2.drawContours(img, [hull], -1, (240, 10, 157), 3)

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    # eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    center = eye_avg + eye_to_mouth * 0.1
    left_chin_point = lm_chin[0]
    right_chin_point = lm_chin[-1]
    left_to_center = center - left_chin_point
    center_to_right = right_chin_point - center
    left_dist = np.hypot(*left_to_center)
    right_dist = np.hypot(*center_to_right)
    ratio = left_dist/right_dist
    # print('ratio', ratio, 'left_dist', left_dist, 'right_dist', right_dist)
    cv2.circle(img, tuple(center.astype(int)), 2, (240, 10, 157), 2)

    if return_ratio:
        return img, ratio
    return img


def compute_random4x4_ratio_cache():
    all_img_files = sorted(glob.glob('./random4x4_samples/random4x4_stylegan_stochastic_noise_*.png'))
    assert len(all_img_files) == 100
    annotated_imgs = []
    img_file_ratio_dict = {}
    predictor = dlib.shape_predictor("./CACHE/shape_predictor_68_face_landmarks.dat")
    for img_file in all_img_files:
        img = dlib.load_rgb_image(img_file)
        lm = get_landmark(img_file, predictor)
        annotated_img, ratio = visualize_landmark(img, lm, return_ratio=True)
        annotated_imgs.append(annotated_img)
        img_file_ratio_dict[img_file] = ratio
    annotated_imgs = torch.from_numpy(np.stack(annotated_imgs, axis=0))
    annotated_imgs = annotated_imgs.permute((0, 3, 1, 2)).float()/255.
    save_image(annotated_imgs, './random4x4_samples/annotated_random4x4_stylegan_stochastic_noise.png', nrow=10)
    torch.save(img_file_ratio_dict, './random4x4_samples/img_file_ratio_dict.pt')
    with open('./random4x4_samples/img_file_ratio_dict.txt', 'w') as out_file:
        for img_file, ratio in img_file_ratio_dict.items():
            out_file.write(f'{img_file} {ratio}\n')


def sort_random4x4_ratio_cache():
    img_file_ratio_dict = torch.load('./random4x4_samples/img_file_ratio_dict.pt')
    img_file_ratio_dict = sorted(img_file_ratio_dict.items(), key=lambda x: x[1])
    all_imgs = []
    for img_file, ratio in img_file_ratio_dict:
        # print(ratio)
        img = dlib.load_rgb_image(img_file)
        all_imgs.append(img)
    all_imgs = torch.from_numpy(np.stack(all_imgs, axis=0))
    all_imgs = all_imgs.permute((0, 3, 1, 2)).float()/255.
    save_image(all_imgs, './random4x4_samples/sorted_random4x4_stylegan_stochastic_noise.png', nrow=10)


def find_closest_ratio(src_ratio, k=1):
    img_file_ratio_dict = torch.load('./random4x4_samples/img_file_ratio_dict.pt')
    all_img_files = list(img_file_ratio_dict.keys())
    all_ratios = list(img_file_ratio_dict.values())
    all_ratios = torch.tensor(all_ratios)
    # print('all_ratios.shape', all_ratios.shape)
    diff_ratios = torch.abs(all_ratios-src_ratio)
    target_ratios, indices = torch.topk(diff_ratios, k, largest=False)
    # print(indices)
    # print(all_ratios[indices])
    target_imgs = []
    for i in indices:
        target_imgs.append(all_img_files[i])
    return target_imgs, indices.tolist()


def my_align_two_imgs(src_img_file, target_img_file=None, prefix='', k=1, debug=False):

    def get_lm_features(lm):
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        center = eye_avg + eye_to_mouth * 0.1
        eye_dist = np.hypot(*eye_to_eye)
        eye_mouth_dist = np.hypot(*eye_to_mouth)

        return center, eye_dist, eye_to_eye, eye_mouth_dist, eye_to_mouth

    def compute_angle(v0, v1):
        unit_v0 = v0 / np.linalg.norm(v0)
        unit_v1 = v1 / np.linalg.norm(v1)
        dot_product = np.dot(unit_v0, unit_v1)
        angle = np.degrees(np.arccos(dot_product))
        # print(unit_v0)
        # print(unit_v1)
        if unit_v0[0] > unit_v1[0]:
            angle = -angle
        return angle

    def compute_matrix(src_lm, target_lm):
        src_center, src_eye_dist, src_eye_to_eye, src_eye_mouth_dist, src_eye_to_mouth = get_lm_features(src_lm)
        target_center, target_eye_dist, target_eye_to_eye, target_eye_mouth_dist, target_eye_to_mouth = get_lm_features(target_lm)

        eye_dist_scale = target_eye_dist / src_eye_dist
        eye_mouth_dist_scale = target_eye_mouth_dist / src_eye_mouth_dist
        scale = (eye_dist_scale + eye_mouth_dist_scale) * 0.5
        # eye_angle = compute_angle(src_eye_to_eye, target_eye_to_eye)
        eye_mouth_angle = compute_angle(src_eye_to_mouth, target_eye_to_mouth)
        # angle = (eye_angle + eye_mouth_angle) * 0.5  # NOTE: avg to make eyes more horizontal
        angle = eye_mouth_angle
        # print('center', src_center)
        # print('angle', angle, 'eye_angle', eye_angle, 'eye_mouth_angle', eye_mouth_angle)
        # print('scale', scale, 'eye_dist_scale', eye_dist_scale, 'eye_mouth_dist_scale', eye_mouth_dist_scale)
        M = cv2.getRotationMatrix2D(tuple(src_center), angle, scale)
        center_offset = target_center - src_center
        M[0, 2] += center_offset[0]
        M[1, 2] += center_offset[1]
        return M

    create_folder('./aligned_imgs')
    create_folder('./tmp')

    prefix = prefix or os.path.splitext(os.path.basename(src_img_file))[0]

    predictor = dlib.shape_predictor("./CACHE/shape_predictor_68_face_landmarks.dat")
    src_img = dlib.load_rgb_image(src_img_file)
    src_lm = get_landmark(src_img_file, predictor)
    annotated_src_img, src_ratio = visualize_landmark(src_img, src_lm, return_ratio=True)
    if target_img_file is None:
        target_img_files, target_indices = find_closest_ratio(src_ratio, k)
    else:
        target_img_files = [target_img_file, ]
        target_indices = None

    all_aligned_filenames = []
    for i, target_img_file in enumerate(target_img_files):
        target_lm = get_landmark(target_img_file, predictor)
        if debug:
            target_img = dlib.load_rgb_image(target_img_file)
            annotated_target_img = visualize_landmark(target_img, target_lm)
            height, width = annotated_target_img.shape[0], annotated_target_img.shape[1]
            log_imgs = np.zeros((height, 3*width, 3)).astype(np.uint8)
            log_imgs[:, :width] = annotated_src_img
            log_imgs[:, width:2*width] = annotated_target_img

        M = compute_matrix(src_lm, target_lm)
        aligned_img = cv2.warpAffine(src_img, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if debug:
            log_imgs[:, 2*width:3*width] = aligned_img
        aligned_img = Image.fromarray(aligned_img)
        target_index = None
        if target_indices is not None:
            target_index = target_indices[i]
        filename = f'./aligned_imgs/{prefix}_{target_index}.png'
        all_aligned_filenames.append(filename)
        aligned_img.save(filename)
        print('Aligned and saved as:', filename)

        if debug:
            log_imgs = Image.fromarray(log_imgs)
            log_imgs.save(f'./tmp/log_imgs_{prefix}_{target_index}.png')
    return all_aligned_filenames


def main(cmd=None, source=None, k=1, debug=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['sample', 'align', ], help='Generate samples or align images.')
    parser.add_argument('--source', help='Image to be aligned.')
    parser.add_argument('--k', type=int, default=1, help='How many pose references to use.')
    parser.add_argument('--debug', action='store_true', help='Show annotated images and landmarks.')

    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args([cmd, f'--source={source}', f'--k={k}', ])
        args.debug = debug

    print(args)

    if args.cmd == 'sample':
        compute_random4x4_ratio_cache()
        sort_random4x4_ratio_cache()
    else:
        all_sources = [x for x in args.source.split(',') if x and os.path.isfile(x)]
        all_aligned_filenames = []
        for source in all_sources:
            all_aligned_filenames.extend(my_align_two_imgs(source, k=args.k, debug=args.debug))
        return all_aligned_filenames


if __name__ == '__main__':
    main()
