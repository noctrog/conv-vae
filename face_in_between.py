#!/bin/python

import argparse
import time
import math as m

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
    parser.add_argument("-i1", type=str, help='input image')
    parser.add_argument("-i2", type=str, help='input image')
    parser.add_argument("-w", type=str, default='', help='Model weights')
    parser.add_argument("-a", type=float, default=0.5, help='interpolation between 0 and 1')

    args = parser.parse_args()

    assert isinstance(args.i1, str)
    assert isinstance(args.i2, str)
    assert isinstance(args.w, str)

    # load model
    dump = torch.load(args.w)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).cuda()
    vae.load_state_dict(dump['state_dict'])
    vae.eval()

    # load image
    img1 = np.asarray(Image.open(args.i1).resize((112, 128))) / 255
    img2 = np.asarray(Image.open(args.i2).resize((112, 128))) / 255
    img1 = np.transpose(img1, [2, 0, 1])
    img2 = np.transpose(img2, [2, 0, 1])
    img1_v = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).cuda()
    img2_v = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).cuda()
    img_v = torch.cat((img1_v, img2_v), 0)
    mu, log_var, output_v = vae.forward(img_v)
    out_img = output_v.detach().squeeze(0).cpu().numpy()

    # obtain median face
    # mu = mu[0] + args.alpha * (mu[1] - mu[0])
    # mu = mu.unsqueeze(0)
    # out_mean = vae.forward_decoder(mu)
    # mean_img = out_mean.detach().squeeze(0).cpu().numpy()

    # plot
    fig = plt.figure()
    # plt.ion()
    # plt.show()
        # while True:
            # alpha = m.cos(time.time()) * 0.5 + 0.5
    alpha = args.a
    mu_ = (mu[0] + alpha * (mu[1] - mu[0])).unsqueeze(0)
    out_mean = vae.forward_decoder(mu_)
    mean_img = out_mean.detach().squeeze(0).cpu().numpy()

    plt.subplot(3, 2, 1, xticks=[], yticks=[])
    plt.imshow(np.transpose(img1, [1, 2, 0]))
    plt.subplot(3, 2, 2, xticks=[], yticks=[])
    plt.imshow(np.transpose(img2, [1, 2, 0]))
    plt.subplot(3, 2, 3, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[0], [1, 2, 0]))
    plt.subplot(3, 2, 4, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[1], [1, 2, 0]))
    plt.subplot(3, 2, 5, xticks=[], yticks=[])
    plt.imshow(np.transpose(mean_img, [1, 2, 0]))
    plt.pause(0.02)
    # plt.draw()
    plt.show()

if __name__ == "__main__":
    main()
