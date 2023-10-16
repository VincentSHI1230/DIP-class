import time
import os
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
from datasets import Dataset
from utils.loss import *
from utils.metrics import pixel_accuracy
from utils import misc
from utils.pytorchtools import EarlyStopping
from torchvision import transforms
import shutil

# 超参设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# crop_size = 512  # 输入裁剪大小
batch_size = 8  # batch size
end_epoch = 50  # 训练的最大epoch，300
iters = 0
pretrained = False  # 是否继续跑
gamma = 0.5  # 学习率下降倍率
early_stop__eps = 1e-2  # 早停的指标阈值
early_stop_patience = 10  # 早停的epoch阈值
initial_lr = 1e-3  # 初始学习率
threshold_lr = 1e-6  # 早停的学习率阈值
weight_decay = 1e-5  # 学习率衰减率
optimizer_type = "adam"  # adam, sgd
scheduler_type = "ReduceLR"  # ReduceLR, StepLR, poly
label_smoothing = 0.01
num_classes = 2  # 分类数
in_channels = 1  # 输入图片通道数
model_number = random.randint(1, 1000)
root_path = "./"
loss_name = "cedice"  # dice, bce, wbce, dual, wdual
model_type = "mobile-unet"

if model_type == "unet":
    from net.baseunet import UNet

    model_name = "{}_{}_{}".format(model_type, loss_name, model_number)
else:
    from net.unet import UNet

    model_name = "{}_{}_{}".format(model_type, loss_name, model_number)

# checkpoint路径,保存模型权重
save_path = os.path.join(root_path, "checkpoint", str(model_number))
if os.path.exists(save_path):
    # 若该目录已存在，则先删除，用来清空数据
    shutil.rmtree(os.path.join(save_path))
os.makedirs(save_path)

# 训练日志
writer = SummaryWriter(
    os.path.join(root_path, "log/bladder/train", model_name + str(int(time.time())))
)
val_writer = SummaryWriter(
    os.path.join(root_path, "log/bladder/val", model_name + str(int(time.time())))
)

# 训练集路径
train_path = r"..\data\Train\Image"
val_path = r"..\data\Validation\Image"


def main():
    # 定义网络
    # net = Baseline(num_classes=num_classes, depth=depth).to(device)
    net = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    # 数据预处理
    # center_crop = joint_transforms.RandomCrop(crop_size)
    center_crop = None
    input_transform = transforms.ToTensor()

    # 训练集加载
    train_set = Dataset(
        train_path, "train", crop=center_crop, transform=input_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=6
    )
    val_set = Dataset(val_path, "val", crop=center_crop, transform=input_transform)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=6
    )
    # val_loader=None

    # 定义损失函数
    if loss_name == "bce_dice":
        criterion = BCE_Dice_Loss(num_classes, weight=(0.7, 0.3)).to(device)
    elif loss_name == "dice":
        criterion = SoftDiceLoss(num_classes).to(device)
    elif loss_name == "cedice":
        criterion = CEdice(num_classes).to(device)
    else:
        criterion = nn.BCELoss().to(device)

    # 定义早停机制
    early_stopping = EarlyStopping(
        early_stop_patience,
        verbose=True,
        delta=early_stop__eps,
        path=os.path.join(save_path, "{}_{}".format(model_type, loss_name)),
    )

    # 定义优化器
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=initial_lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    # 定义学习率衰减策略
    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    elif scheduler_type == "ReduceLR":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=gamma, patience=4, threshold=1e-2
        )
    else:
        scheduler = None

    # 继续跑
    if pretrained:
        state_dict = torch.load("./checkpoint/752/last.pth")
        net.load_state_dict(state_dict["model"])
        scheduler.load_state_dict(state_dict["scheduler"])
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict["epoch"] + 1
    else:
        start_epoch = 1

    train(
        train_loader,
        val_loader,
        net,
        criterion,
        optimizer,
        scheduler,
        None,
        early_stopping,
        start_epoch,
        end_epoch,
        iters,
    )


def train(
    train_loader,
    val_loader,
    net,
    criterion,
    optimizer,
    scheduler,
    warm_scheduler,
    early_stopping,
    start_epoch,
    num_epoches,
    iters,
):
    epoch = 0
    for epoch in range(start_epoch, num_epoches + 1):
        st = time.time()
        train_class_dices = []
        train_recalls = []
        train_precisions = []
        train_accs = []
        val_class_dices = []
        val_recalls = []
        val_precisions = []
        val_accs = []
        train_losses = []
        val_losses = []

        if num_classes == 2:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            loss_weight = torch.as_tensor([0.1, 1], device=device)
        else:
            loss_weight = None

        # 训练模型
        net.train()
        for batch, ((inputs, mask), file_name) in enumerate(train_loader, 1):
            X = inputs.to(device)
            y = mask.to(device)
            optimizer.zero_grad()
            output = net(X)
            a = y.long().squeeze(1)
            # 计算损失函数
            loss = criterion(output, a, loss_weight)
            # 梯度回传
            loss.backward()
            # 更新优化器
            optimizer.step()
            iters += 1
            train_losses.append(loss.item())

            output = torch.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)

            output = output.cpu().detach()
            mask = mask.squeeze(1).detach()

            recall, PA_b, acc, precision, dice, pixel_labeled = pixel_accuracy(
                output, mask
            )
            if pixel_labeled == 0:
                recall = 0.0
                precision = 0.0
                dice = 0.0
            else:
                train_class_dices.append(dice)
                train_recalls.append(recall)
                train_precisions.append(precision)
            train_accs.append(acc)

            string_print = "epoch: {}, iters: {}, loss: {:.4}, dice:{:.4},recall:{:.4},precision:{:.4},time: {:.2}".format(
                epoch, iters, loss.data.cpu(), dice, recall, precision, time.time() - st
            )
            misc.log(string_print)
            st = time.time()

        train_loss = np.average(train_losses)
        train_class_dice = np.average(train_class_dices)
        train_recall = np.average(train_recalls)
        train_precision = np.average(train_precisions)
        train_acc = np.average(train_accs)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("loss", train_loss, epoch)
        writer.add_scalar("dice(f1)", train_class_dice, epoch)
        writer.add_scalar("precision", train_precision, epoch)
        writer.add_scalar("recall", train_recall, epoch)
        writer.add_scalar("acc", train_acc, epoch)

        print(
            "epoch {}/{}, train_loss: {:.4}, dice(f1):{:.4},acc:{:.4},recall:{:.4},precision:{:.4}".format(
                epoch,
                num_epoches,
                train_loss,
                train_class_dice,
                train_acc,
                train_recall,
                train_precision,
            )
        )

        # # 验证模型
        net.eval()
        for val_batch, ((inputs, mask), file_name) in tqdm(enumerate(val_loader, 1)):
            val_X = inputs.to(device)
            val_y = mask.to(device)
            pred = net(val_X)
            a = val_y.long().squeeze(1)
            val_loss = criterion(pred, a, loss_weight)
            val_losses.append(val_loss.item())

            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().detach()
            mask = mask.squeeze(1).detach()

            (
                valrecall,
                valPA_b,
                valacc,
                valprecision,
                valdice,
                valpixel_labeled,
            ) = pixel_accuracy(pred, mask)

            if valpixel_labeled != 0:
                val_class_dices.append(valdice)
                val_recalls.append(valrecall)
                val_precisions.append(valprecision)
            val_accs.append(valacc)

        val_loss = np.average(val_losses)
        val_class_dice = np.average(val_class_dices)
        val_recall = np.average(val_recalls)
        val_precision = np.average(val_precisions)
        val_acc = np.average(val_accs)

        val_writer.add_scalar("loss", val_loss, epoch)
        val_writer.add_scalar("dice(f1)", val_class_dice, epoch)
        val_writer.add_scalar("recall", val_recall, epoch)
        val_writer.add_scalar("acc", val_acc, epoch)
        val_writer.add_scalar("precision", val_precision, epoch)

        print(
            "val_loss:{:.4}, dice(f1):{:.4},recall:{:.4},precision:{:.4},acc:{:.4}".format(
                val_loss, val_class_dice, val_recall, val_precision, val_acc
            )
        )

        print("lr: {}".format(optimizer.param_groups[0]["lr"]))
        # 判断早停
        early_stopping(val_class_dice, net, optimizer, scheduler, epoch)
        if early_stopping.early_stop or optimizer.param_groups[0]["lr"] < threshold_lr:
            print("Early stopping")
            break

        # 根据val_class_dice判断是否更新学习率
        scheduler.step(val_class_dice)
        # 保存模型参数
        save_file = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(save_file, os.path.join(save_path, "last.pth"))

    print("----------------------------------------------------------")
    print("save epoch {}".format(early_stopping.save_epoch))
    print("stoped epoch {}".format(epoch))
    print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
