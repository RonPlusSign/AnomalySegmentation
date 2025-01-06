from PIL import Image, ImageOps
import random
from transform import Relabel, ToLabel, Colorize
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, RandomResizedCrop, ColorJitter
import cv2
import numpy as np
import PIL.ImageEnhance as ImageEnhance

#Augmentations - different function implemented to perform random augments on both image and target
class ERFNetTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

class BiSeNetTrainTransform(object):
    def __init__(self):
        scales=[0.75, 2.]
        cropsize=[1024, 1024]

        self.trans_func = Compose([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, input, target):
        input = self.trans_func(input)
        target = self.trans_func(target)
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target
    
class BiSeNetEvalTransform(object):
    def __call__(self, input, target):
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target
    
class ENetTransform(object):
    def __init__(self, height=512, width=1024):
        self.height = height
        self.width = width

        # Define the image and label transforms
        self.image_transform = Compose([Resize((self.height, self.width)), ToTensor()])
        self.label_transform = Compose([Resize((self.height, self.width), Image.NEAREST), ToTensor()])
    
    def __call__(self, input, target):
        input = self.image_transform(input)
        target = self.label_transform(target)

        target = ToLabel()(target).squeeze(1)
        target = Relabel(255, 19)(target)
        return input, target
