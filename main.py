import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import DatasetFromFolder
from model import SRCNN

import numpy as np
import cv2
import skimage.measure

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--nb_epochs', type=int, default=400)
parser.add_argument('--cuda', action='store_true')
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Parameters
BATCH_SIZE = 4
NUM_WORKERS = 1 # on Windows, set this variable to 0

trainset = DatasetFromFolder("/export/liuzhe/data/dataset/DIV2K3/DIV2K_train_HR", zoom_factor=args.zoom_factor)
testset = DatasetFromFolder("/export/liuzhe/data/dataset/DIV2K3/DIV2K_val_HR", zoom_factor=args.zoom_factor)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model = SRCNN().to(device)
criterion = nn.L1Loss()
mse = nn.MSELoss()
# optimizer = optim.Adam(  # we use Adam instead of SGD like in the paper, because it's faster
#     [
#         {"params": model.conv1.parameters(), "lr": 0.001},
#         {"params": model.conv2.parameters(), "lr": 0.001},
#         {"params": model.conv3.parameters(), "lr": 0.001},
#         {"params": model.conv4.parameters(), "lr": 0.0001},
#     ], lr=0.001,
# )
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)   # 固定步长衰减  多步长衰减 指数衰减 余弦退火衰减

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.5 ** (epoch // 10))
    return lr

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


for epoch in range(args.nb_epochs):

    # Train
    lr = adjust_learning_rate(optimizer, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    epoch_loss = 0
    for iteration, batch in enumerate(trainloader):
        input, target, target_hr, input_1 = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        optimizer.zero_grad()

        model.initialize()
        out = model(input, input_1)      # out = [4,8,32,32], target = [4,8,32,32]

        # out = channel_8_1(out)
        # target = channel_8_1(target)    # out = [4,1,32,32], target = [4,1,32,32]

        # print("11111")
        # print(out.shape)
        # print(target.shape)

        # # 保存8通道hr、lr图像
        # hr_8 = target.cpu().detach().data.numpy()
        # sr_8 = out.cpu().detach().data.numpy()
        # # print(sr_8)
        # print(np.max(sr_8))
        # print(np.mean(sr_8))
        # print(np.max(hr_8))
        # print(np.mean(hr_8))
        # # print(hr_8)
        # exit(-1)

        loss = criterion(out, target)

        loss = loss.requires_grad_()

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # StepLR.step()
        # lr = StepLR.get_lr()
        # print(epoch, StepLR.get_lr()[0])

        epoch_loss += loss.item()

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")
    # Save model
    torch.save(model, f"model_{epoch}.pth")

    # Test
    model.eval()
    avg_psnr = 0
    num = 0
    with torch.no_grad():
        for batch in testloader:
            num = num + 1
            input, target, not_8 = batch[0].to(device), batch[1].to(device), batch[3].to(device)

            # model = torch.load('model_399.pth').to(device)
            out = model(input, not_8)

            # out = out.cpu().detach().data.numpy()
            # out = out.clip(0, 1)
            # out = torch.from_numpy(np.ascontiguousarray(out)).float()
            # out = out.cuda()

            # out = channel_8_1(out)
            # target = channel_8_1(target)  # out = [4,1,32,32], target = [4,1,32,32]

            # 保存8通道hr、lr图像
            # hr_8 = target.cpu().detach().data.numpy()
            # sr_8 = out.cpu().detach().data.numpy()
            # hr_8 = np.transpose(hr_8[0], [1, 2, 0])  # [96,96,8]
            # sr_8 = np.transpose(sr_8[0], [1, 2, 0])  # [96,96,8]

            # psnr = skimage.measure.compare_psnr(sr_8, hr_8, 255)
            # print(psnr)


            # import skimage.measure
            # ssim = skimage.measure.compare_ssim(sr_8, hr_8 , data_range=1, multichannel=True)
            # psnr = skimage.measure.compare_psnr(sr_8, hr_8 , data_range=1)
            # print(psnr)
            # exit(-1)

            # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr.png', hr_8 )
            # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr.png', sr_8 )

            # for i in range(8):
            #     cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr_{}.png'.format(i), hr_8[:, :, i] * 255)
            #     cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr_{}.png'.format(i), sr_8[:, :, i] * 255)
            # exit(-1)

            # loss = mse(out, target)

            out = channel_8_1(out)
            target = channel_8_1(target)  # out = [4,1,32,32], target = [4,1,32,32]
            # target = target * 255

            out = out.cpu().detach().data.numpy()[0, 0, :, :]
            target = target.cpu().detach().data.numpy()[0, 0, :, :]

            psnr = skimage.measure.compare_psnr(out, target, 255)
            # print(psnr)

            # import imageio
            # imageio.imwrite('/export/liuzhe/program2/SRCNN/data/'
            #                 +str(num)+'out.png',out)
            # imageio.imwrite('/export/liuzhe/program2/SRCNN/data/'
            #                 + str(num) + 'target.png', target)

            # psnr = 10 * log10(1 / loss.item())
            # print(psnr)
            avg_psnr += psnr
    print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")
    # adjust_learning_rate(optimizer, epoch)
    # StepLR.step()
