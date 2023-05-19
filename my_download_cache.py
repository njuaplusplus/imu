#!/usr/bin/env python3
# coding=utf-8
import os
import subprocess


ALL_CACHE = {
    # conf_mask.pt
    'CACHE/conf_mask.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EfGCarPZZ-BGmE4RnZpVze4BVogJVdI3K46JDJzJsqcU5g?e=evkn75&download=1',
    # pre-sampled ws
    'CACHE/stylegan_sample_z_stylegan_celeba_partial256_0.7_8_all_ws.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/Eee2Fvs7269DoZ8bRVKbBjEBY6bi0z02eLc6ApOiTc-wwA?e=Bz7Zzy&download=1',
    # pre-generated noises
    'random4x4_samples/random4x4_stylegan_stochastic_noise.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EawmjsYsF75KkWII4m3zph4BwDx0DWt_fdRFkVrZf9meQQ?e=vRSuBi&download=1',
    # resnet50 on vggface2
    'CACHE/resnet50_scratch_dag.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EZOXU_L8CQdHvWdWnRfV7F4BCE-JGamMjKYwBWuPk5pyVQ?e=Hmzguu&download=1',
    # vgg16 on vggface
    'CACHE/vgg_face_dag.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EXTGdHk8fnZCriclmVjRFeIBif04VtUlKOYSFF9a1fh08A?e=sAaAQi&download=1',
    # vgg16bn on vggface
    'CACHE/vgg_m_face_bn_dag.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EeN5LkNMRS5LsHFVJoBQE40BojD0tA1RgkyaoSn6Z8GXZw?e=NyX5zOi&download=1',
    # SphereFace on CASIA
    'CACHE/sphere20a_20171020.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EfDfgGbvTTFPixaKLWcvhq8B0OlSDRdB_QRrc2Li45659A?e=LHtmVd&download=1',
    # dlib landmark file
    'CACHE/shape_predictor_68_face_landmarks.dat': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EVaJR2pE_eRGrY6SPsPcVhMBm368LSKX2nFfsjJ7lLHYtw?e=CVCgzg&download=1',
    # The following face models are downloaded from TencentYoutuResearch/Adv-Makeup
    'Pretrained_FR_Models/facenet.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EToE-Xzi9u1Lkkm-W5a5cy0BFM5kxQW8xNviQpin96ChQA?e=MdG8m4&download=1',
    'Pretrained_FR_Models/ir152.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/ETPEvS9xz-BOir9YWN19RdwBUMIlmayI517MiDK4VZWZSg?e=w20lJK&download=1',
    'Pretrained_FR_Models/irse50.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/ES0LWEH3WklNkp6mSHQylLEBD-0UwSSrDlyYczVHnsjBzg?e=v6MNSf&download=1',
    'Pretrained_FR_Models/mobile_face.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/ER64FqKKpC1JuN0vHi-7-DUBsKF1mGhg_tIN3Aqi0lIHlQ?e=Hp3qH5&download=1',
}


def download(filename, url):

    def create_folder(folder):
        if os.path.exists(folder):
            assert os.path.isdir(folder), 'it exists but is not a folder'
        else:
            os.makedirs(folder)

    if not os.path.exists(filename):
        print('Downloading', filename)
        dirname = os.path.dirname(filename)
        if dirname:
            create_folder(dirname)
        subprocess.call(['wget', '--quiet', '-O', filename, url])


def main():
    for filename, url in ALL_CACHE.items():
        download(filename, url)


if __name__ == '__main__':
    main()
