#根据标注信息把GT检测框画在图片上
import os
import cv2
import xml.etree.ElementTree as ET
from utils.utils import get_classes

#表示当前文件的父目录，不随着调用主函数的路径变化而变化
#os.path.dirname("__file__")表示当前文件路径，不随调用函数路径的变化而变化
#os.path.pardir表示父目录，但是不能用“..”代替
# parent_dir=os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
parent_dir=os.path.abspath(os.path.dirname("__file__"))
classes_path=os.path.join(parent_dir,'model_data/VOCfire_class.txt')
# classes_path = r'D:\git\mobilenet-yolov4-pytorch\model_data\VOCfire_class.txt'

classes, _      = get_classes(classes_path)

# annotations_root = r'D:\git\mobilenet-yolov4-pytorch\VOCdevkit\VOCfire\Annotations'
annotations_root=os.path.join(parent_dir,'VOCdevkit/VOCfire/Annotations')

#从xml文件中读取gt数据
def get_gt(gt_path):
    '''
    从xml标注文件中读取gt数据
    :param gt_path: 标注文件地址
    :return: gts，shape=(n,4),每一行为left,top,right,bottom
    '''
    gts=[]
    tree = ET.parse(gt_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue

        xmlbox = obj.find('bndbox')
        b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))]

        gts.append(b)
    return gts


#根据gt文件，在对应的图片上画上gt
def draw_gt(image_path,gt_path):
    image=cv2.imread(image_path)
    gts=get_gt(gt_path)
    for gt in gts:
        left,top,right,bottom=gt
        cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),1)
    cv2.imwrite(image_path,image)


#给文件夹中的图片画上gt
def plot_gt(images_root):
    image_list = os.listdir(images_root)
    for image in image_list:
        image_path = os.path.join(images_root, image)
        annotation = image.replace('.jpg', '.xml')
        annotation = annotation.replace('.png', '.xml')
        annotation_path = os.path.join(annotations_root, annotation)
        draw_gt(image_path, annotation_path)


if __name__=='__main__':
    images_root=r'D:\git\mobilenet-yolov4-pytorch\fire_test_out_0.4_0911'
    plot_gt(images_root)
