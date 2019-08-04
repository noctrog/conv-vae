#!/bin/python

import argparse

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import model

def normalize(img, mean, std):
    return (img - mean) / std

def unnormalize(img, mean, std):
    return img * std + mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help='input image')
    parser.add_argument("-p", default=False, action='store_true', help='pipe image')
    parser.add_argument("-o", type=str, default='', help='output image')
    parser.add_argument("-w", type=str, default='', help='Model weights')

    args = parser.parse_args()

    assert isinstance(args.i, str)
    assert isinstance(args.o, str)
    assert isinstance(args.w, str)
    assert args.p and args.o == ''

    # load model
    dump = torch.load(args.w)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).cuda()
    vae.load_state_dict(dump['state_dict'])
    vae.eval()

    # load image
    img = np.asarray(Image.open(args.i)) / 255
    img = np.transpose(img, [2, 0, 1])
    img_norm = normalize(img, 0.5, 0.5)
    img_v = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).cuda()
    output_v = vae.forward_no_epsilon(img_v)
    output = output_v.detach().squeeze(0).cpu().numpy()
    out_img = unnormalize(output, 0.5, 0.5)

    # plot
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.transpose(img, [1, 2, 0]))
    plt.subplot(2, 1, 2)
    plt.imshow(np.transpose(out_img, [1, 2, 0]))
    plt.show()


if __name__ == "__main__":
    main()
