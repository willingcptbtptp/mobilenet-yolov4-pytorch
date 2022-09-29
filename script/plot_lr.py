#画出学习率曲线
import matplotlib.pyplot as plt
from nets.yolo_training import get_lr_scheduler


optimizer_type='sgd'
lr_decay_type='cos'
Freeze_Epoch=50
UnFreeze_Epoch=350

freeze_batch_size=8
unfreeze_batch_size=4

Init_lr = 16e-2
Min_lr = Init_lr * 0.001
nbs = 64  # 该batch对应的初始学习率为0.01
lr_limit_max = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
lr_limit_min = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4

#unfreeze阶段的学习率
batch_size=freeze_batch_size
#自适用初始学习率，根据batch成比例缩放，但是不能超过[lr_limit_min， lr_limit_max]范围
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)

#自适用最小学习率，是Init_lr_fit的0.01倍
Min_lr_fit  = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

lr_list=[]
for epoch in range(UnFreeze_Epoch):
    lr = lr_scheduler_func(epoch)
    if epoch >= Freeze_Epoch:
        batch_size = unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        lr = lr_scheduler_func(epoch)
    lr_list.append(lr)


plt.plot(range(UnFreeze_Epoch),lr_list)