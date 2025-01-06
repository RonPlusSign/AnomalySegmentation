from PIL import Image, ImageOps
import random
from transform import Relabel, ToLabel, Colorize
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, RandomHorizontalFlip, ToTensor
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
    def __init__(self, img_mean, img_std, img_scale, crop_size):
        self.img_mean = img_mean
        self.img_std = img_std
        self.img_scale = img_scale
        self.crop_size = crop_size

    def __call__(self, img, gt):
        img = self.color_jitter(img)
        img, gt = self.random_mirror(img, gt)
        img, gt = self.random_scale(img, gt, self.img_scale)
        img = self.normalize(img, self.img_mean, self.img_std)
        img, gt = self.random_crop(img, gt, self.crop_size)
        img = img.transpose(2,0,1) #C * H * W

        return img, gt
        
    def random_mirror(img, gt):
        if random.random() >= 0.5:
            img = cv2.flip(img, 1)
            gt = cv2.flip(gt, 1)

        return img, gt

    def random_scale(img, gt, scales):
        scale = random.choice(scales)
        sh = int(img.shape[0] * scale)
        sw = int(img.shape[1] * scale)
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

        return img, gt

    def normalize(img, mean, std):
        mean = np.array(mean)
        std = np.array(std)

        img = img.astype(np.float32) / 255.0
        img = img - mean
        img = img / std

        return  img

    def random_crop(self, img, gt, crop_size):
        h, w = img.shape[:2]
        crop_h, crop_w = crop_size[0], crop_size[1]

        if h > crop_h:
            x = random.randint(0, h - crop_h + 1)
            img = img[x:x + crop_h, :, :]
            gt = gt[x:x + crop_h, :]

        if w > crop_w:
            x = random.randint(0, w - crop_w + 1)
            img = img[:, x:x + crop_w, :]
            gt = gt[:, x:x + crop_w]

        img,_ = self.shape_pad(img, crop_size, 0)
        gt,_ = self.shape_pad(gt, crop_size, 255)

        return img, gt

    def color_jitter(img, brightness=0.5, contrast=0.5, saturation=0.5):
        img = Image.fromarray(img)
        brightness_range = [max(1 - brightness, 0), 1 + brightness]
        contrast_range = [max(1 - contrast, 0), 1 + contrast]
        saturation_range = [max(1 - saturation, 0), 1 + saturation]

        r_brightness = random.uniform(brightness_range[0], brightness_range[1])
        r_contrast = random.uniform(contrast_range[0], contrast_range[1])
        r_saturation = random.uniform(saturation_range[0], saturation_range[1])

        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)

        return np.asarray(img)

    def shape_pad(img, shape, value):
        margin = np.zeros(4, np.uint32)

        pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
        pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

        margin[0] = pad_height // 2
        margin[1] = pad_height // 2 + pad_height % 2
        margin[2] = pad_width // 2
        margin[3] = pad_width // 2 + pad_width % 2

        img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                                cv2.BORDER_CONSTANT, value=value)

        return img, margin

class BiSeNetEvalTransform(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        img = self.normalize(img, self.img_mean, self.img_std)
        img = img.transpose(2,0,1) #C * H * W

        return img, gt
    
    def normalize(img, mean, std):
        mean = np.array(mean)
        std = np.array(std)

        img = img.astype(np.float32) / 255.0
        img = img - mean
        img = img / std

        return  img