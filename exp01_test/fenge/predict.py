import os
import cv2
import torch
import shutil
from utils.loss import *
from PIL import Image
from utils.metrics import Evaluator
import matplotlib.pyplot as plt
from torchvision import transforms

model_type = "munet"
in_channels=1
crop_size = 512
val_input_transform = transforms.ToTensor()
zhuan= transforms.ToPILImage()
num_classes = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_type == "unet":
    from net.baseunet import UNet
else:
    from net.unet import UNet

val_path = r'/HOME/scz0490/run/yy/p/ddsm-data/test'
names=open(r'/HOME/scz0490/run/yy/p/ddsm-data/test/train-yuan.txt','r').readlines()


net = UNet(in_channels=in_channels,num_classes=num_classes).to(device)
#读入训练好的模型参数
net.load_state_dict(torch.load("./checkpoint/5.9/70250/mobile-unet_cedice_14.pth")['model'])

net.eval()

def auto_val(net):
    # 效果展示图片数
    save_path = './results'
    if os.path.exists(save_path):
        # 若该目录已存在，则先删除，用来清空数据
        shutil.rmtree(os.path.join(save_path))
    img_path = os.path.join(save_path, 'images')
    gt_path = os.path.join(save_path, 'gt')
    pre_path = os.path.join(save_path, 'pre')
    full_path = os.path.join(save_path, 'full-images')
    full_pre = os.path.join(save_path, 'full-pre')
    os.makedirs(img_path)
    os.makedirs(gt_path)
    os.makedirs(pre_path)
    os.makedirs(full_path)
    os.makedirs(full_pre)
    right=0
    t=crop_size
    for i in range(0,len(names)):
        name = names[i].replace('\n','')
        # name='P_00016_LEFT_CC.png'
        img = cv2.imread(os.path.join(val_path, 'images-clahe',name),0)
        mask = cv2.imread(os.path.join(val_path, 'labels',name),0)
        h,w=img.shape[:2]
        xt,yt = int(h/2048)+1,int(w/2048)+1
        img1 = cv2.resize(img,(t*yt,t*xt),interpolation=cv2.INTER_CUBIC)
        new = Image.new('L',(yt*t,xt*t),(0))
        for x in range(xt):
            for y in range(yt):
                img2 = img1[t*x:(x+1)*t,t*y:(y+1)*t]#剪切
                img3 = val_input_transform(img2)
                X = img3.unsqueeze(1).to(device)
                pred = net(X)
                pred = torch.softmax(pred,dim=1)
                output = torch.argmax(pred, dim=1)
                output = zhuan(output.float()) #tensor-pil
                new.paste(output,(t*y,t*x))#拼回去
                # plt.imshow(new,cmap='gray')
                # plt.show()
        newpred = np.array(new)
        newpred = cv2.resize(newpred,(w,h),interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((30,30),np.uint8)
        imgr=img.copy()
        imgr=cv2.merge([imgr,imgr,imgr])
        closing = cv2.morphologyEx(newpred, cv2.MORPH_CLOSE, kernel)
        _, contours_gt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, contours_pred, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgr, contours_gt,-1,(0,255,0),5) #(0,0,255)为轮廓颜色，绿色，1为轮廓线粗细
        cv2.drawContours(imgr, contours_pred,-1,(255,0,0),5) #(0,0,255)为轮廓颜色，红色，1为轮廓线粗细
        cv2.imwrite(os.path.join(full_path, name),imgr)
        cv2.imwrite(os.path.join(full_pre, name),closing)
        recall,PA_b,acc,precision,dice,sd = pixel_accuracy(closing, mask)
        print('picture:{},dice(f1):{:.4},precision(精确率): {:.4},PA_mass(recall):{:.4}'.
              format(name,dice,precision, recall))
        k=0
        for n in range(len(contours_pred)):
            # 筛选面积较大的连通区，阈值为20000
            cnt = contours_pred[n]
            area = cv2.contourArea(cnt)
            if area > 1000:
                x,y,w1,h1=cv2.boundingRect(cnt)
                y0=y+h1//2
                x0=x+w1//2
                if x0<112:
                    x0=112
                elif x0+112>w:
                    x0=w-112
                if y0<112:
                   y0=112
                elif y0+112>h:
                    y0=h-112
                if h1<224:
                    if w1<224:
                       imgq=img[y0-112:y0+112,x0-112:x0+112]
                       gt=mask[y0-112:y0+112,x0-112:x0+112]
                       pred=closing[y0-112:y0+112,x0-112:x0+112]
                    else:
                        imgq=img[y0-112:y0+112,x:x+w1]
                        gt=mask[y0-112:y0+112,x:x+w1]
                        pred=closing[y0-112:y0+112,x:x+w1]
                else:
                    if w1<224:
                       imgq=img[y:y+h1,x0-112:x0+112]
                       gt=mask[y:y+h1,x0-112:x0+112]
                       pred=closing[y:y+h1,x0-112:x0+112]
                    else:
                        imgq=img[y:y+h1,x:x+w1]
                        gt=mask[y:y+h1,x:x+w1]
                        pred=closing[y:y+h1,x:x+w1]
                name1=name[:-4]+'{}.png'.format(k)
                k=k+1
                cv2.imwrite(os.path.join(img_path, name1),imgq)
                cv2.imwrite(os.path.join(pre_path, name1),pred)
                cv2.imwrite(os.path.join(gt_path, name1),gt)
        # print(name)
if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)
    # t=['last.pth','last-150.pth','last-228.pth']
    # net = UNet(in_channels=in_channels,num_classes=num_classes).to(device)
    # for i in t:
    #     net.load_state_dict(torch.load(os.path.join('../checkpoint/55038',i), map_location='cpu')['model'])
    net.eval()
    auto_val(net)