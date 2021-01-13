import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io

parser = argparse.ArgumentParser(description='SRCNN run parameters')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

img = Image.open(args.image).convert('YCbCr')
img = img.resize((int(img.size[0]*args.zoom_factor), int(img.size[1]*args.zoom_factor)), Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
y, cb, cr = img.split()
img_to_tensor = transforms.ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
# print(input.shape)

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
print(device)
model = torch.load(args.model).to(device)
input = input.to(device)

out = model(input, input)
out *= 255


def hr_convert(self, hr):
    # [1,32,32]
    img = hr
    img = (np.array(img)[0, 1] * 255).astype("uint8")
    # print(img)
    # exit(-1)

    # img_new = img

    img_new = (img.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
    img = img.astype("uint8")
    img_new = img ^ img_new  # 异或运算

    [h, w] = img_new.shape[0], img_new.shape[1]

    image = np.empty((1, 8, h, w), dtype=np.uint8)  # 存 余数

    for i in range(8):
        image[0, i, :, :] = img_new % 2  # 转格雷码8维图像
        img_new = img_new // 2
    # print(image)
    # print(np.mean(image))

    x_finally = torch.from_numpy(np.ascontiguousarray(image)).float()  # [8,32,32]
    # image = torch.from_numpy(np.ascontiguousarray(image)).float()
    # x_finally = image.cuda()
    return x_finally

def str_reverse1(s):
    return s[::-1]
def channel_8_1(sr):
    # [4, 8, 32, 32]
    x_0 = (sr.cpu().detach().data.numpy())  # x转numpy
    x_0 = x_0.astype('float64')
    x_mid = np.empty([x_0.shape[0], 1, x_0.shape[2], x_0.shape[3]], dtype='uint8')
    x_mid8 = np.empty([x_0.shape[0], 8, x_0.shape[2], x_0.shape[3]], dtype='uint8')
    x_bin = np.zeros([x_0.shape[1], x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [8, h, w]
    x_finally = np.zeros([x_0.shape[0], 1, x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [1,1,512,512]

    h, w = x_0.shape[2], x_0.shape[3]
    for p in range(x_0.shape[0]):
        x_mid[p, :, :, :] = x_0[p, 0, :, :] * 1 + x_0[p, 1, :, :] * 2 + x_0[p, 2, :, :] * 4 + x_0[p, 3, :, :] * 8 + x_0[p, 4, :,:] * 16 + x_0[p, 5,:,:] * 32 + x_0[p, 6,:,:] * 64 + x_0[p, 7,:,:] * 128
        for i in range(8):
            x_mid8[p, i, :, :] = x_mid[p, 0, :, :] % 2  # 转格雷码8维图像
            x_mid[p, 0, :, :] = x_mid[p, 0, :, :] // 2

        img1 = x_mid8[p, :, :, :]
        # 十进制格雷码转二进制的十进制数
        for i in range(h):
            for j in range(w):
                lst = []
                for s in range(8):
                    lst.append(str(img1[s, i, j]))
                n = ''.join(lst)  # str从低位到高位
                n = str_reverse1(n)  # str从高位到低位

                result = ''
                for q in range(8):  # 格雷码转二进制码
                    if q != 0:
                        temp = 1
                        if result[q - 1] == n[q]:
                            temp = 0
                        result += str(temp)
                    else:
                        result += str(n[0])

                result = str_reverse1(result)  # 从低位到高位
                for m in range(8):
                    x_bin[m, i, j] = result[m] # [8, h ,w] 解码后的二进制码

        x_temp = x_bin[0, :, :]*1 + x_bin[1, :, :]*2 + x_bin[2, :, :]*4 + x_bin[3, :, :]*8 + x_bin[4, :, :]*16 + x_bin[5, :, :]*32 + x_bin[6, :, :]*64 + x_bin[7, :, :]*128
        # x_temp = x_bin[0, :, :] * 128 + x_bin[1, :, :] * 64 + x_bin[2, :, :] * 32 + x_bin[3, :, :] * 16 + x_bin[4, :, :] * 8 + x_bin[5, :, :] * 4 + x_bin[6, :, :] * 2 + x_bin[7, :, :] * 1
        x_finally[p, :, :, :] = x_temp

    # img_1 = x_1  # [1, 512, 512]     (y_decode.png)

    # img_2 = img_1

    '''# 十进制格雷码转二进制的十进制数
    img_2 = (img_1.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
    img = img_1.astype("uint8")
    img_2 = img ^ img_2  # 异或运算   [1, 512, 512]'''

    # x_finally[:, 0, :, :] = img_2

    x_finally = torch.from_numpy(np.ascontiguousarray(x_finally)).float()  # [4,1,32,32]
    # x_finally /= 255
    x_finally = x_finally.cuda()
    # print(x_finally)

    return x_finally


out = channel_8_1(out)
#srimg = np.transpose(out.cpu().detach().data.numpy()[0], (1, 2, 0))
out = out.cpu()
out_img_y = out[0].detach().numpy()
print(out_img_y)
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)

out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')


# plt.imshow(out_img_y)
# plt.show()

out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')  # we merge the output of our network with the upscaled Cb and Cr from before
                                                                    # before converting the result in RGB
out_img.save(f"zoomed_{args.image}")
# plt.imshow(out_img)
# plt.show()
