import os
import xml.etree.ElementTree as ET
import shutil

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    # ------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    # -------------------------------------------------------------------------------------------------------------------#
    map_mode = 0
    # --------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    # --------------------------------------------------------------------------------------#
    # classes_path    = 'model_data/voc_classes.txt'
    classes_path = 'model_data/VOCfire_class.txt'

    # -------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    # -------------------------------------------------------#
    map_vis = False

    # -------------------------------------------------------#
    #   用于计算map的数据集voc目录，计算map的时候使用VOCXX/ImageSets/Main/test.txt中的数据
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit/VOCfire_city'

    # -------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    # -------------------------------------------------------#
    map_out_path = 'map_out'

    # --------------------------------------------------------------------------------------#
    #   MINOVERLAP是根据predict和GT的IOU判断检测框是否为label正的阈值，每个MINOVERLAP对应一个PR曲线
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    # --------------------------------------------------------------------------------------#
    MINOVERLAP = 0.5  # 源码中是0.5，有点苛刻
    # MINOVERLAP = 0.4

    # ---------------------------------------------------------------------------------------------------------------#
    #   score_threhold是划分predict confidence，判定predict box是否为预测正的阈值，confidence大于该阈值为预测正
    #   每个score_threshold对应的是PR曲线上的一个点
    #   score_threhold是确定P-R曲线上某一点，输出改点的P、R、F1
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    # ---------------------------------------------------------------------------------------------------------------#
    # 如果predict的recall很低，我们可以适当的降低score_threshold，以提高predict出现的概率
    # 例如score_threhold = 0.4
    score_threhold = 0.5

    '''
    下面这部分参数，传入yolo.py中是用来更新yolo.py中对应的参数，构造model。
    为了保证此时计算的map与训练时候方法相同，这里的超参数也与训练时候保持一致
    '''

    model_path = 'logs/VOCfire_city_adam_model2_log20220924/ep190-loss0.027-val_loss0.027.pth'
    # ---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    anchors_path = 'yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # ---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    # ---------------------------------------------------------------------#
    input_shape = [416, 416]

    # ---------------------------------------------------------------------#
    #   检测网络所使用的主干
    # ---------------------------------------------------------------------#
    backbone = 'mobilenetv2'

    # --------------------------------------------------------------------------------------#
    #   predicr阶段的截断阈值，也就是NMS中截断阈值
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #   该值会更新yolo.py中的confidence
    # --------------------------------------------------------------------------------------#
    confidence = 0.001  # 源代码中是0.001,但是训练的时候用的是0.05

    # --------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   该值一般不调整。
    #   该值会更新yolo,py中的nms_iou
    # --------------------------------------------------------------------------------------#
    nms_iou = 0.5

    # ---------------------------------------------------------------------#
    #   从NMS之后的predict中筛选出置信度最高的max_boxes个检测框（所有类别一起排序）
    #   训练的时候用的100，可以大大加快计算map的速度
    #   因为confidence+NMS筛选之后平均每张图还会生成数千个检测框
    # ---------------------------------------------------------------------#
    max_boxes = 100

    # 是否同比例缩放输入图片，False表示直接缩放
    # 经过实验发现，False或True得到的map根据数据集不同，互有高低
    # 但是为了保证该此时计算map的步骤与val时候一样（val的时候），这里选择True，等比例缩放
    letterbox_image = True

    image_ids = open(os.path.join(VOCdevkit_path, "ImageSets/Main/val.txt")).read().strip().split()

    # 删除map_out_path文件夹，因为其中存有上一次计算map所保留的detect_result和ground_truth，会对本次map计算造成影响
    shutil.rmtree(map_out_path)  # 会直接把文件夹删除，因此后面需要新建该文件夹

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(model_path=model_path, anchors_path=anchors_path, anchors_mask=anchors_mask,
                    input_shape=input_shape, backbone=backbone, confidence=confidence,
                    nms_iou=nms_iou, max_boxes=max_boxes, letterbox_image=letterbox_image)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
