import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import shutil

from utils.utils import get_classes

#处理训练数据集，从VOC数据集中提取两个txt文件2007_train.txt，2007_val.txt，
#存放训练和验证数据的路径以及GT信息
'''
VOC数据集的目录组成
Annotations：标签文件夹，内部每个图片一个xml文件
ImageSets：
    Main：存放索引的文件夹，内部包括train，val，test等文件，这些文件存储的是对应的图片集索引
JPEGImages：图片文件夹，内部存储上万张图片
'''
#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt,
#   我新增加的：mode=1还要把test图片拷贝到img_test中，方便后序的测试
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 0
#-------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
#-------------------------------------------------------------------#
# classes_path        = 'model_data/voc_classes.txt'
classes_path        = 'model_data/VOCfire_class.txt'

img_test=r'./img_test'

#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    =0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'

# VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
#对于自己的数据集，需要把年份2007换成VOC后面的字符
VOCdevkit_sets  = [('fire_city', 'train'), ('fire_city', 'val')]
classes, _      = get_classes(classes_path)

#-------------------------------------------------------#
#   统计目标数量
#-------------------------------------------------------#
photo_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(len(classes))

#读取指定图片的标注文件，把gt信息写到list_file（2007_train.txt、2007_val.txt）中
#有的img不存在gt信息，为了方便删除这些文件，我把该函数改成不直接把gt信息写入，
# 而是返回待写入的标注信息（str格式）
def convert_annotation(year, image_id, list_file):

    #这是我新加的代码
    gt_str=''

    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))

        # 这是我新加的代码
        gt_str+=" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)

        #这是原来的代码
        # list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1

    #这是我新加的代码
    return gt_str
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        if os.path.exists(img_test):
            # 清空img_test文件夹，该函数会顺便把文件夹也删除
            shutil.rmtree(img_test)
        os.mkdir(img_test)
        print("清空test_img文件夹，并重新拷贝test图片.")
        print("Generate txt in ImageSets.")
        year=VOCdevkit_sets[0][0]
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC%s/Annotations'%(year))
        jpgfilepath     = os.path.join(VOCdevkit_path, 'VOC%s/JPEGImages'%(year))
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main'%(year))
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []

        total_img=os.listdir(jpgfilepath)
        total_img_no_suffix=[i[:-4] for i in total_img]

        for xml in temp_xml:
            #判断为xml文件的同时还要满足有对应的jpg文件
            if xml.endswith(".xml") and xml[:-4] in total_img_no_suffix:
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)
                #把test图片加入到img_test文件夹中
                img_name=total_xml[i][:-4]+'.jpg'
                img_path=os.path.join(jpgfilepath,img_name)
                dst_path=os.path.join(img_test,img_name)
                shutil.copy(img_path,dst_path)


        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0

        #读取ImageSets/Main中的train.txt和val.txt文件，获取图片路径和gts
        for year, image_set in VOCdevkit_sets:
            # image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').readlines()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                # 这是原来的代码
                # list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))
                #
                # convert_annotation(year, image_id, list_file)

                #这是我修改后的代码
                image_id=image_id.strip()
                gt_str=convert_annotation(year, image_id.strip(), list_file)

                #把所有不含有classes中指定目标的样本全部筛选掉
                #最初是为了筛选cat_and_dog中cat和dog的样本
                #但这样会把所有不含检测目标的样本筛掉，也把hard_negative example筛掉了
                #因此还是需要把所有背景样本都加入
                # if gt_str=='':
                #     continue

                #源码是绝对路径，为了能在远程跑，改成相对路径
                # path_str='%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id)
                path_str = '%s/VOC%s/JPEGImages/%s.jpg' % (VOCdevkit_path, year, image_id)
                list_file.write(path_str)
                list_file.write(gt_str)

                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")
