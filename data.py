from os import listdir
from os.path import join

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

import numpy as np
import torch

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

CROP_SIZE = 48

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, zoom_factor):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        #torchvision.transforms.ToTensor()起到的作用是把PIL.Image或者numpy.narray数据类型转变为torch.FloatTensor类型，shape是C*H*W，数值范围缩小为[0.0, 1.0]
        crop_size = CROP_SIZE - (CROP_SIZE % zoom_factor) # Valid crop size
        self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                      transforms.Resize(crop_size//zoom_factor),  # subsampling the image (half size)
                                      transforms.Resize(crop_size, interpolation=Image.BICUBIC),  # bicubic upsampling to get back the original size 
                                      transforms.ToTensor()])
        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size), # since it's the target, we keep its original quality
                                       transforms.ToTensor()])

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        
        # input = input.filter(ImageFilter.GaussianBlur(1)) 
        input = self.input_transform(input)
        target = self.target_transform(target)
        input_1 = input.clone()
        target_hr = target.clone()
        # print(input.shape)
        # print(type(input))
        # exit(-1)
        # input = input * 255
        # print(input)

        target = self.hr_convert(target)  # [8, 32, 32]
        input = self.hr_convert(input)
        # target = target * 255

        return input, target, target_hr, input_1

    def __len__(self):
        return len(self.image_filenames)

    def hr_convert(self, hr):
        # [1,32,32]
        img = hr
        img = (np.array(img)[0] * 255).astype("uint8")
        # print(img)
        # exit(-1)

        # img_new = img

        img_new = (img.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
        img = img.astype("uint8")
        img_new = img ^ img_new  # 异或运算

        [h, w] = img_new.shape[0], img_new.shape[1]

        image = np.empty((8, h, w), dtype=np.uint8)  # 存 余数

        for i in range(8):
            image[i, :, :] = img_new % 2  # 转格雷码8维图像
            img_new = img_new // 2
        # print(image)
        # print(np.mean(image))

        x_finally = torch.from_numpy(np.ascontiguousarray(image)).float()  # [8,32,32]

        # x_finally = image.cuda()
        return x_finally
