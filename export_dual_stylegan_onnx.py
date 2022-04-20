import os
import wget
import numpy as np
import types
import argparse
from argparse import Namespace

import sys
sys.path.append('./')

from util import save_image, load_image, visualize
from model.dualstylegan import DualStyleGAN
from model.sampler.icp import ICPTrainer
from model.encoder.psp import pSp

import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

def parse_args(parser):
    '''
    Parse commandline args
    '''
    # model_dir / style / align_face / 
    parser.add_argument('--model_dir', type=str, required=True,
                        help='full path to the models dir')
    parser.add_argument('--style', type=str, required=True, default='cartoon',
                        help="style name, default is cartoon ")

def run_alignment(image_path, args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_dir, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image


# for test
class parser() :
    model_dir = '/DualStyleGAN/checkpoint',
    style_img = '/DualStyleGAN/data/cartoon/images/train/Cartoons_00167_01.jpg'
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    
if __name__ == '__main__':
    
    args = parser()
    
    transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

    test_input_img = '/content/DualStyleGAN/data/content/unsplash-rDEOVtE7vOs.jpg'
    test_input_img = run_alignment(test_input_img, args)
    test_input_img = transform(run_alignment(test_input_img)).unsqueeze(dim=0).to(args.device)