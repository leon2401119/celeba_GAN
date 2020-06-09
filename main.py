from __future__ import print_function
# %matplotlib inline
import os
import random
import torch
# import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import argparse
# from IPython.display import HTML
from Model import *



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")




def common_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.00008, type=float)

    return parser.parse_args()


def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):

    # load saved params
    # generator.load_state_dict(torch.load('gen_params.pkl'))
    # discriminator.load_state_dict(torch.load('dis_params.pkl'))


    # Each epoch, we have to go through every data in dataset
    total_loss_g = []
    total_loss_d = []
    iter = 0
    plot_axis = []
    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):

            # initialize gradient for network
            # send the data into device for computation
            optimizer_d.zero_grad()
            data[0] = data[0]*2 - 1 # rescale to [-1,1]
            data_x = data[0].to(device)

            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            real_out = discriminator(data_x)
            real_label = torch.ones(real_out.shape)
       
            # Using Fake data, other steps are the same.
            # Generate a batch fake data by using generator
            noise = np.random.normal(0, 1, size=[128, 100, 1, 1])   # are 512 fake samples too much (edited to 128)
            noise = torch.from_numpy(noise)
            noise = noise.type(torch.cuda.FloatTensor)
            gen_data = generator(noise.to(device))

            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            fake_out = discriminator(gen_data)
            fake_label = torch.zeros(fake_out.shape)
           
            # Update your network
            # loss_g = nn.functional.binary_cross_entropy(torch.ones(fake_out.size()).to(device), fake_out.to(device))
            # loss_g.backward()

            real_loss = nn.functional.binary_cross_entropy(real_out.to(device), real_label.to(device))
            fake_loss = nn.functional.binary_cross_entropy(fake_out.to(device), fake_label.to(device))
            loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer_d.step()

            # Record your loss every iteration for visualization
            if iter % 50 == 0:
                total_loss_d.append(loss_d.item())
           
            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if i % 1 == 0:
                print('ver'+ str(epoch) + ' discriminator loss:', loss_d.item())

            # if i>50:
                # mean = 0
                # for x in total_loss_d[i-5:i]:
                    # mean += x
                # mean /= 5
                # if loss_d.item() > mean and loss_d.item() < mean * 1.5:
                    # break
            # if (i>5 and loss_d.item()<0.0001)   or   (i>100):
                # break

            optimizer_g.zero_grad()
            noise = np.random.normal(0, 1, size=[128, 100, 1, 1])
            noise = torch.from_numpy(noise)
            noise = noise.type(torch.cuda.FloatTensor)
            gen_data = generator(noise.to(device))
            fake_out = discriminator(gen_data)
            fake_label = torch.ones(fake_out.shape)
            loss_g = nn.functional.binary_cross_entropy(fake_out.to(device), fake_label.to(device))
            loss_g.backward()
            optimizer_g.step()
            if iter % 50 == 0:
                total_loss_g.append(loss_g.item())
            print('ver' + str(epoch) + ' generator loss:', loss_g.item())

            if iter % 50 == 0:
                plot_axis.append(iter)
            iter += 1

            # Remember to save all things you need after all batches finished!!!



        fig = plt.gcf()
        sample = np.random.normal(0, 1, size=[4, 100, 1, 1])
        sample = torch.from_numpy(sample)
        sample = sample.type(torch.cuda.FloatTensor)
        generator.eval()
        img = ((generator(sample.to(device)).detach().cpu())+1)/2 #rescale to [0,1]
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1)
            # plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            ax.imshow(img[i].permute(1, 2, 0))
        plt.savefig('Generator_v' + str(epoch) + '_Sample.png')


        torch.save(generator.state_dict(), 'gen_params_v' + str(epoch) + '.pkl')
        torch.save(discriminator.state_dict(), 'dis_params_v' + str(epoch) + '.pkl')
        generator.train()


    xtick = [0,3000,6000,9000,12000,15000]
    # plot curve
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Generator and Discriminator Loss During Training')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.plot(plot_axis, total_loss_g, label='Generator')
    ax.plot(plot_axis, total_loss_d, label='Discriminator')
    ax.set_xticks(xtick)
    ax.legend()
    plt.savefig('Learning Curve')



def main(args):
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor() ])
    dataset = dset.ImageFolder(args.dataroot, transform)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    


    # Create the generator and the discriminator()
    # Initialize them 
    # Send them to your device
    generator = Generator(ngpu)
    discriminator = Discriminator(ngpu)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = optim.Adam(generator.parameters(), args.lr, betas=(0.5,0.5))
    optimizer_d = optim.Adam(discriminator.parameters(), args.lr, betas=(0.5,0.5))
    criterion = nn.CrossEntropyLoss()
    
    
    # Start training~~
    
    train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, args.num_epochs)
    


if __name__ == '__main__':
    args = common_arg_parser()
    main(args)