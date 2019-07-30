#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

import argparse

import model

def unnormalize_img(img):
    img = img / 2 + 0.5
    return img


def kl_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)

def r_loss(y_train, y_pred):
    r_loss = torch.mean((y_train - y_pred) ** 2, dim=(1,2,3))
    return r_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--data_dir", help="Folder where the data is located")
    parser.add_argument("--epochs", type=int, help='Number of times to iterate the whole dataset')
    parser.add_argument("--visual_every", default=200, type=int, help='Display faces every n batches')
    parser.add_argument("--z_dim", type=int, default=200, help='Dimensions of latent space')
    parser.add_argument("--r_loss_factor", type=float, default=10000.0, help='r_loss factor')
    parser.add_argument("--lr", default=0.002, help='Learning rate')
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size')
    parser.add_argument("--load", type=str, help='Load pretrained weights')
    args = parser.parse_args()

    # data where the images are located
    data_dir = args.data_dir
    assert isinstance(data_dir, str)
    assert isinstance(args.epochs, int)
    assert isinstance(args.visual_every, int)
    assert isinstance(args.z_dim, int)
    assert isinstance(args.r_loss_factor, float)
    assert isinstance(args.lr, float)
    assert isinstance(args.batch_size, int)

    # use CPU or GPU
    device = torch.device("cuda" if args.cuda else "cpu")

    # prepare data
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # create model
    input_shape = next(iter(trainloader))[0].shape
    vae = model.VAE(input_shape[-3:], args.z_dim).to(device)
    print(vae)          # print for feedback

    # load previous weights (if any)
    if args.load is not '':
        vae.load_state_dict(torh.load(args.load))
        print("Weights loaded: {}".format(args.load))

    # create tensorboard writer
    writer = SummaryWriter(comment='-' + 'VAE' + str(args.z_dim))

    optimizer = optim.Adam(vae.parameters(), lr=0.002)

    # generate random points in latent space so we can see how the network is training
    latent_space_test_points = np.random.normal(scale=1.0, size=(6, args.z_dim))
    latent_space_test_points_v = torch.Tensor(latent_space_test_points).to(device)

    batch_iterations = 0
    training_losses = []
    vae.train()
    for e in range(args.epochs):
        epoch_loss = []
        for images, labels in trainloader:
            images_v = images.to(device)

            optimizer.zero_grad()

            mu_v, log_var_v, images_out_v = vae(images_v)
            r_loss_v = r_loss(images_out_v, images_v)
            kl_loss_v = kl_loss(mu_v, log_var_v)
            print(r_loss_v.size())
            print(kl_loss_v.size())
            loss = kl_loss_v + r_loss_v * args.r_loss_factor
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            if batch_iterations % args.visual_every == 0:
                # print loss
                print("Batch: {}\tLoss: {}".format(batch_iterations + e * len(trainloader) / args.batch_size, loss.item()))
                writer.add_scalar('loss', loss.item(), batch_iterations)


            batch_iterations = batch_iterations + 1

        else:
            training_losses.append(np.mean(epoch_loss))
            if min(training_losses) == training_losses[-1]:
                torch.save(vae.state_dict(), 'vae_weights-' + str(args.z_dim) + '.dat')

            vae.eval()

            generated_imgs_v = vae.forward_decoder(latent_space_test_points_v)

            writer.add_image('preview-1', unnormalize_img(generated_imgs_v[0].detach().cpu().numpy()), batch_iterations)
            writer.add_image('preview-2', unnormalize_img(generated_imgs_v[1].detach().cpu().numpy()), batch_iterations)
            writer.add_image('preview-3', unnormalize_img(generated_imgs_v[2].detach().cpu().numpy()), batch_iterations)
            writer.add_image('preview-4', unnormalize_img(generated_imgs_v[3].detach().cpu().numpy()), batch_iterations)
            writer.add_image('preview-5', unnormalize_img(generated_imgs_v[4].detach().cpu().numpy()), batch_iterations)
            writer.add_image('preview-6', unnormalize_img(generated_imgs_v[5].detach().cpu().numpy()), batch_iterations)

            vae.train()


if __name__ == "__main__":
    main()
