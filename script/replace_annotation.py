import os
import shutil


path_ori=r'D:\git\mobilenet-yolov4-pytorch\VOCdevkit\VOCfire_city\Annotations'
path_dst=r'D:\git\mobilenet-yolov4-pytorch\VOCdevkit\VOCfire\Annotations'

xml_list=os.listdir(path_ori)
for xml in xml_list:
    xml_ori=os.path.join(path_ori,xml)
    xml_dst=os.path.join(path_dst,xml)
    shutil.copy(xml_ori,xml_dst)
