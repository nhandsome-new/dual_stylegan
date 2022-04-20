import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import os
import sys
import wget
import argparse
import numpy as np
from argparse import Namespace

from util import save_image, load_image, visualize
from model.dualstylegan import DualStyleGAN
from model.sampler.icp import ICPTrainer
from model.encoder.psp import pSp

sys.path.append('./')

MODEL_PATHS = {
    "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
    "cartoon-G": {"id": "1exS9cSFkg8J4keKPmq2zYQYfJYC5FkwL", "name": "generator.pt"},
    "cartoon-N": {"id": "1JSCdO0hx8Z5mi5Q5hI9HMFhLQKykFX5N", "name": "sampler.pt"},
    "cartoon-S": {"id": "1ce9v69JyW_Dtf7NhbOkfpH77bS_RK0vB", "name": "refined_exstyle_code.npy"},
    "caricature-G": {"id": "1BXfTiMlvow7LR7w8w0cNfqIl-q2z0Hgc", "name": "generator.pt"},
    "caricature-N": {"id": "1eJSoaGD7X0VbHS47YLehZayhWDSZ4L2Q", "name": "sampler.pt"},
    "caricature-S": {"id": "1-p1FMRzP_msqkjndRK_0JasTdwQKDsov", "name": "refined_exstyle_code.npy"},
    "anime-G": {"id": "1BToWH-9kEZIx2r5yFkbjoMw0642usI6y", "name": "generator.pt"},
    "anime-N": {"id": "19rLqx_s_SUdiROGnF_C6_uOiINiNZ7g2", "name": "sampler.pt"},
    "anime-S": {"id": "17-f7KtrgaQcnZysAftPogeBwz5nOWYuM", "name": "refined_exstyle_code.npy"},
    "arcane-G": {"id": "15l2O7NOUAKXikZ96XpD-4khtbRtEAg-Q", "name": "generator.pt"},
    "arcane-N": {"id": "1fa7p9ZtzV8wcasPqCYWMVFpb4BatwQHg", "name": "sampler.pt"},
    "arcane-S": {"id": "1z3Nfbir5rN4CrzatfcgQ8u-x4V44QCn1", "name": "exstyle_code.npy"},
    "comic-G": {"id": "1_t8lf9lTJLnLXrzhm7kPTSuNDdiZnyqE", "name": "generator.pt"},
    "comic-N": {"id": "1RXrJPodIn7lCzdb5BFc03kKqHEazaJ-S", "name": "sampler.pt"},
    "comic-S": {"id": "1ZfQ5quFqijvK3hO6f-YDYJMqd-UuQtU-", "name": "exstyle_code.npy"},
    "pixar-G": {"id": "1TgH7WojxiJXQfnCroSRYc7BgxvYH9i81", "name": "generator.pt"},
    "pixar-N": {"id": "18e5AoQ8js4iuck7VgI3hM_caCX5lXlH_", "name": "sampler.pt"},
    "pixar-S": {"id": "1I9mRTX2QnadSDDJIYM_ntyLrXjZoN7L-", "name": "exstyle_code.npy"},    
    "slamdunk-G": {"id": "1MGGxSCtyf9399squ3l8bl0hXkf5YWYNz", "name": "generator.pt"},
    "slamdunk-N": {"id": "1-_L7YVb48sLr_kPpOcn4dUq7Cv08WQuG", "name": "sampler.pt"},
    "slamdunk-S": {"id": "1Dgh11ZeXS2XIV2eJZAExWMjogxi_m_C8", "name": "exstyle_code.npy"},     
}


# import models

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--model_dir', type=str, default='/content/DualStyleGAN/checkpoint',
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--data_dir', type=str, default='/content/DualStyleGAN/data',
                        help='data_dir')
    parser.add_argument('--style_type', type=str, default='cartoon',
                        help='style_type')
    parser.add_argument('--generator_path', type=str, default='cartoon', default='/content/DualStyleGAN/checkpoint/cartoon/generator.pt',
                        help='style_type')
    parser.add_argument('--device', type=str, default='cpu',
                        help='DEVICE ')
    parser.add_argument('--input_image', type=str, default='/content/DualStyleGAN/data/content/unsplash-rDEOVtE7vOs.jpg',
                        help='full path to the input image')
    parser.add_argument('--if_align_face', type=bool, default=True,
                        help='align face with pretrained model')
    parser.add_argument('--output', type=str, default='/content/drive/MyDrive/fusic/202204/onnx',
                        help='onnt output path')
    return parser

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

class pSpEncoder(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder.to(args.device)
        
    def forward(self, input_image):
        img_rec, instyle = self.encoder.forward(input_image, randomize_noise=False, return_latents=True, 
                                        z_plus_latent=True, return_z_plus_latent=True, resize=False)
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        
        return img_rec, instyle

class StyleTransfer(nn.Module):
    def __init__(self, generator, args):
        # load DualStyleGAN model
        generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
        ckpt = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        generator = generator.to(args.device)
        
        # exstyle : mixed style to latent representation
        # generator : latent representation to image
        # exstyles : pre-trained latent representation
        self.exstyler = generator.generator.style
        self.generator = generator
        self.exstyles = np.load(os.path.join(args.model_dir, args.style_type, MODEL_PATHS[args.style_type+'-S']["name"]), allow_pickle='TRUE').item()
   
    def forward(self, instyle, style_id=21, interp_weights=[0.6]*7+[1]*11):
        # pre-trained style + extracted style of input image
        style_id = style_id
        stylename = list(self.exstyles.keys())[style_id]
        latent = torch.tensor(self.exstyles[stylename]).repeat(2,1,1).to(self.device)
        # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
        latent[1,7:18] = instyle[0,7:18]
        
        # style to latent
        exstyle = self.generator.generator.style(
                    latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
        
        img_gen, _ = self.generator([instyle.repeat(2,1,1)], exstyle, z_plus_latent=True, 
                    truncation=0.7, truncation_latent=0, use_res=True, interp_weights=interp_weights)
        
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        
        return img_gen

def main():
    # init configures
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 export to TRT')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    
    model_dir = args.model_dir
    style_type = args.style_type
    input_image = args.input_image
    device = args.device
    
    
    # ################################
    # # pSp Encoder to ONNX
    # ################################
    # # load encoder
    # model_path = os.path.join(model_dir, 'encoder.pt')
    # ckpt = torch.load(model_path, map_location='cpu')
    # opts = ckpt['opts']
    # opts['checkpoint_path'] = model_path
    # opts = Namespace(**opts)
    # opts.device = device
    # encoder = pSp(opts)
    # encoder.eval()
    # encoder = encoder.to(device)
    
    # # Set the encoder model into inference mode
    # # options
    # psp_encoder = pSpEncoder(encoder, args)

    # # dummy input image
    # input_img = torch.rand((1,3,256,256)).float().to(device)
    
    # # export encoder onnx
    # opset_version = 11
    # psp_encoder.eval()
    # torch.onnx.export(psp_encoder, input_img, args.output+"/"+"encoder.onnx",
    #                   opset_version=opset_version,
    #                   input_names=["input_img"],
    #                   output_names=["img_rec", "instyle"],
    #                   do_constant_folding=True,
    #                 #   dynamic_axes={"input_img": {0: "input_batch"}}
    # )
    
    ################################
    # Style Transfer
    ###############################
    # # style extrector
    # exstyles = np.load(os.path.join(model_dir, style_type, MODEL_PATHS[style_type+'-S']["name"]), allow_pickle='TRUE').item()
    
    # # load DualStyleGAN model
    # generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    # ckpt = torch.load(os.path.join(model_dir, style_type, 'generator.pt'), map_location=lambda storage, loc: storage)
    # generator.load_state_dict(ckpt["g_ema"])
    # generator = generator.to(device)

    # dummy input image
    # (36, 512)
    # style_id = 26
    # stylename = list(exstyles.keys())[style_id]
    # latent = torch.tensor(exstyles[stylename]).repeat(2,1,1).to(device)
    # # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
    # latent[1,7:18] = latent[0,7:18]
    # print(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2]).shape)
    input_style = torch.rand((36, 512)).float().to(device)
    
    model = StyleTransfer()
    opset_version = 11
    model.eval()
    torch.onnx.export(model, input_style, args.output+"/"+"style_transfer.onnx",
                        opset_version=opset_version,
                        input_names=["instyle"],
                        output_names=["style_latent"],
                        do_constant_folding=True,
    )
    
    
    # model1.eval()
    # print(f'MODEL 1 input:{input_img.shape} and output size is...')
    # output1 = model1(input_img)
    # print(output1.shape)
    
    # opset_version = 12
    # model1.eval()
    # torch.onnx.export(model1, input_img, args.output+"/"+"encoder_encoder.onnx",
    #                   opset_version=opset_version,
    #                   )
    
    ################################
    # test psp_encoder : decoder
    #   TESTING : PASS
    ################################
    # class model2(nn.Module):
    #     def __init__(self, model):
    #             super().__init__()
    #             self.model = model
    #     def forward(self, x):
    #             return self.model([x],
    #                         input_is_latent=False,
    #                         randomize_noise=False,
    #                         return_latents=True,
    #                         z_plus_latent=True)
    
    # model2 = model2(psp_encoder.encoder.decoder)
    # input2 = torch.rand((1, 18, 512)).float().to(device)
    # model2.eval()
    # print(f'MODEL 2 input:{input2.shape} and output size is...')
    # output2 = model2(input2)
    # print(output2)
    
    # opset_version = 11
    # model2.eval()
    # torch.onnx.export(model2, input2, args.output+"/"+"encoder_decoder.onnx",
    #                   do_constant_folding=True,
    #                   opset_version=opset_version,
    #                   )
    
    
if __name__ == '__main__':
    main()