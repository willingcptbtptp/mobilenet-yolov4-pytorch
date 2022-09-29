import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
from pyinstrument import Profiler


#评估模型时间
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/log/prof'),
        record_shapes=True,
        with_stack=True)

def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, batch_size,gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0,
                  mAP_mode='mAP0.5'):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()

    #评估模型时间
    prof.start()
    for iteration, batch in enumerate(gen):

        if iteration >= (1 + 1 + 3) * 2:
            break

        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            loss_value_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs)):
                #计算得到的loss已经在batch_size上平均了
                # （实际计算的是时候loss除以对应的obj_mask和noobj_mask）
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(images)

                loss_value_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        #源代码中是直接累加每个batch的loss，最后记录以及画图得到的结果都是每个batch_size的平均loss
        #这就可能会让loss随着bs的增大而增大，让人产生迷惑
        #因此我把它改记录每张图的平均loss，20220921
        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss':'{:.5f}'.format(loss / (iteration + 1)),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

        # 评估模型时间
        prof.step()

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            loss_value_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': '{:.5f}'.format( val_loss / (iteration + 1))})
            pbar.update(1)
        # 评估模型时间
        prof.step()

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')


        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        #评估训练效果，计算map
        eval_callback.on_epoch_end(epoch + 1, model_train,mAP_mode)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.5f || Val Loss: %.5f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    # 评估模型时间
    prof.stop()