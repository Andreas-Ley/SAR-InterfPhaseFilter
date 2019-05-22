import PhaseNet.net as net
import torch
import numpy as np
import scipy.signal as signal
from PIL import Image


def save2ChannelImg(img, filename, scale):
    image = np.stack((img[0,:,:], img[1,:,:], np.zeros((img.shape[1], img.shape[2]))), axis=-1)
    im = Image.fromarray(np.minimum(np.maximum(image * scale + 127, 0), 255).astype('uint8'), 'RGB')
    im.save(filename)

def save3ChannelImg(img, filename, scale):
    image = np.stack((img[0,:,:] * scale + 127, img[1,:,:] * scale + 127, img[2,:,:] * 256), axis=-1)
    im = Image.fromarray(np.minimum(np.maximum(image, 0), 255).astype('uint8'), 'RGB')
    im.save(filename)

net = net.Network()


input = np.load("0000_corrupt.npz")['corrupt']

save2ChannelImg(input, "input.png", 127)

input = torch.from_numpy(input[np.newaxis,:,:,:])
output = net.forward(input)
output = output.detach().numpy()[0,:,:,:]

save3ChannelImg(output, "output.png", 100)

