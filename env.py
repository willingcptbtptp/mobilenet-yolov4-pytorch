import torch
import platform


gpu_num=torch.cuda.device_count()
platform_version=platform.version()
python_version=platform.python_version()

print('platform_version:%s'%platform_version)
print('python_version:python %s'%python_version)
print('torch version:%s'%torch.__version__)
print('cuda version:CUDA %s'%torch.version.cuda)
print('cuda is avaliable:%s'%torch.cuda.is_available())
print('gpu num:%d'%gpu_num)
for i in range(gpu_num):
    print('gpu %d ---->gpu name:%s'%(i,torch.cuda.get_device_name(i)))



