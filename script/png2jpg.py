from PIL import Image
import os

def png2jpg(png_path):
    in_path=png_path
    out_path=in_path[:-4]+'.jpg'
    im = Image.open(in_path)
    im = im.convert('RGB')
    im.save(out_path)


path=r'D:\CNN_data\data_set\fire\my_fire_dataset\fire-like\JPEGImages'
png_list=os.listdir(path)
for png in png_list:
    png_path=os.path.join(path,png)
    png2jpg(png_path)