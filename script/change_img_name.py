#网上下载的数据集中文件的命名不够规范，有的文件名含有空格，这样会对后序文件的读取等有影响
#这里把空格全部删掉

import os


jpg_root=r'D:\git\mobilenet-yolov4-pytorch\VOCdevkit\VOCfire\JPEGImages'
xml_root=r'D:\git\mobilenet-yolov4-pytorch\VOCdevkit\VOCfire\Annotations'

for root in [jpg_root,xml_root]:
    path_list=os.listdir(root)
    for path in path_list:
        old_path=os.path.join(root,path)
        path_no_space=path.replace(' ','')  #替换空格
        new_path=os.path.join(root,path_no_space)
        if old_path!=new_path:
            os.rename(old_path,new_path)
