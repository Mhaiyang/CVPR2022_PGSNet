"""
 @Time    : 2021/10/16 15:12
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : cvpr2022
 @File    : infer.py
 @Function:
 
"""
import time
import datetime
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
from tqdm import tqdm

from config import *
from misc import *
from model.pgsnet import PGSNet

torch.manual_seed(2022)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = '/home/mhy/tpami2022/results/220602/11'
check_mkdir(results_path)
ckpt_path = '/home/mhy/tpami2022/ckpt'
exp_name = 'PGSNet'
args = {
    'snapshot': '180',
    'scale': 416,
    'save_results': True,
}

print(torch.__version__)

image_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
aolp_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
])
dolp_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    # ('RGBP-Glass', testing_root),
    ('Demo', demo_root),
])

results = OrderedDict()

def main():
    net = PGSNet(backbone_path1, backbone_path2, backbone_path3, backbone_path4).cuda(device_ids[0])

    # print('Load snapshot {}.pth for testing'.format(args['snapshot']))
    # net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    # print('Load {} succeed!'.format(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.load_state_dict(torch.load(pgsnet_ckpt_path))
    print('Load {} succeed!'.format(pgsnet_ckpt_path))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')
            aolp_path = os.path.join(root, 'aolp')
            dolp_path = os.path.join(root, 'dolp')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.tiff')]
            img_list.sort(key=lambda x: int(x.split('_')[0]))
            img_list = ['%04d'%int(x.split('_')[0]) + x[-4:] for x in img_list]
            for img_name in tqdm(img_list):
                # print(img_name)
                image = Image.open(os.path.join(image_path, img_name + '.tiff')).convert('RGB')
                aolp = Image.open(os.path.join(aolp_path, img_name[:-4] + '_aolp.tiff')).convert('RGB')
                aolp = reorder(aolp)
                dolp = Image.open(os.path.join(dolp_path, img_name[:-4] + '_dolp.tiff')).convert('RGB')
                dolp = reorder(dolp)

                w, h = image.size
                image_var = Variable(image_transform(image).unsqueeze(0)).cuda(device_ids[0])
                aolp_var = Variable(aolp_transform(aolp).unsqueeze(0)).cuda(device_ids[0])
                dolp_var = Variable(dolp_transform(dolp).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                prediction = net(image_var, aolp_var, dolp_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.float().squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(
                        os.path.join(results_path, exp_name, img_name[:-4] + '_mask.png'))

            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.1f} ms".format(name, mean(time_list) * 1000))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


if __name__ == '__main__':
    main()
