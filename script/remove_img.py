import os


#根据JPEGImages中的图片，去筛选annotation中放入标注数据
path=r'D:\git\mobilenet-yolov4-pytorch\VOCdevkit\VOCfire_forest'
jpg_path=os.path.join(path,'JPEGImages')
annotation_path=os.path.join(path,'Annotations')

jpg_list=os.listdir(jpg_path)
annotation_left_list=[]
for jpg in jpg_list:
    annotation=jpg.replace('.jpg','.xml')
    annotation_left_list.append(annotation)


annotation_list_ori=os.listdir(annotation_path)
for annotation in annotation_list_ori:
    if annotation not in annotation_left_list:
        annotation_remove=os.path.join(annotation_path,annotation)
        os.remove(annotation_remove)



