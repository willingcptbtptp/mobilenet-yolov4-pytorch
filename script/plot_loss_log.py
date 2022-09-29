import os
import scipy.signal
import matplotlib.pyplot as plt


def test_ploy_log():
    loss_path=r'D:\git\mobilenet-yolov4-pytorch\logs\VOCfire_city_log20220916\loss_2022_09_16_00_35_42\epoch_loss.txt'
    f=open(loss_path,'r')
    loss_list=f.readlines()
    y=[float(i.strip()) for i in loss_list]
    x=range(len(y))
    plt.semilogy(x,y)

def loss_plot(losses,val_loss):
    iters = range(len(losses))

    plt.figure()
    # 把之前的plt的y轴改成log坐标
    # plt.plot(iters, losses, 'red', linewidth = 2, label='train loss')
    # plt.plot(iters, val_loss, 'coral', linewidth = 2, label='val loss')
    plt.semilogy(iters, losses, 'red', linewidth=2, label='train loss')
    plt.semilogy(iters, val_loss, 'coral', linewidth=2, label='val loss')
    try:
        if len(losses) < 25:
            num = 5
        else:
            num = 15

        # 把原来的plot的y轴改为log坐标
        # plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        # plt.plot(iters, scipy.signal.savgol_filter(val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        plt.semilogy(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
        plt.semilogy(iters, scipy.signal.savgol_filter(val_loss, num, 3), '#8B4513', linestyle='--',
                     linewidth=2,
                     label='smooth val loss')
    except:
        pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.show()

    plt.savefig( "epoch_loss.png")

    plt.cla()
    plt.close("all")


def get_loss(loss_path):
    f = open(loss_path, 'r')
    loss_list = f.readlines()
    loss = [float(i.strip()) for i in loss_list]
    return loss


def main():
    train_loss_path=r'D:\git\mobilenet-yolov4-pytorch\logs\VOCfire_city_log20220915\loss_2022_09_15_07_47_59\epoch_loss.txt'
    val_loss_path=r'D:\git\mobilenet-yolov4-pytorch\logs\VOCfire_city_log20220915\loss_2022_09_15_07_47_59\epoch_val_loss.txt'
    train_loss=get_loss(train_loss_path)
    val_loss=get_loss(val_loss_path)
    loss_plot(train_loss,val_loss)


if __name__=='__main__':
    main()
