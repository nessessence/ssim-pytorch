import argparse
import os
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torchvision import transforms
from modules.ssim import ssim
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataSet


def main():
    args = parser.parse_args()


    if args.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids  # new add by gu
        cudnn.benchmark = True
        gpu_ids = sorted([int(gpu_id.strip() ) for  gpu_id in args.gpu_ids.split(',')])
        torch.cuda.set_device(torch.device('cuda',gpu_ids[0]))


    img_size = args.img_size
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )

    dataset1 = CustomDataSet(args.dataset_path1,transform=transform)
    dataset2 = CustomDataSet(args.dataset_path2,transform=transform)

    batch_size = args.batch_size
    dataloader1 = DataLoader(dataset1 , batch_size=batch_size, shuffle=False,  num_workers=4, drop_last=False)
    dataloader2 = DataLoader(dataset2 , batch_size=batch_size, shuffle=False,  num_workers=4, drop_last=False)
    
    ssim_score = []
    for imgs1,imgs2 in tqdm(zip(dataloader1,dataloader2)):
        if args.device == 'cuda': 
            imgs1 = imgs1.cuda()
            imgs2 = imgs2.cuda()
        ssim_score += [ssim(imgs1, imgs2, size_average=False)]
    ssim_score = torch.cat(ssim_score).mean()
    print(f"total avg ssim score: {ssim_score}")

    with open(args.output_log,'w') as f:
        f.write(f"total avg ssim score: {ssim_score}")
    print(f"result has been saved to {args.output_log}")

    print("the process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d1','--dataset_path1', type=str, default='/home/nessessence/DDPM/stylegan2-pytorch/datasets/CelebA-HQ-img')
    parser.add_argument('-d2','--dataset_path2', type=str, default='/data/nessessence/DDPM/projected_output/W_PLUS/inversed_imgs/')
    parser.add_argument('-gpu','--gpu_ids', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda','cpu'])
    parser.add_argument('-s','--img_size', type=int, default=128)
    parser.add_argument('-b','--batch_size', type=int, default=2048)
    parser.add_argument('-o','--output_log', type=str, default='output/wplus_ssim_result.txt')
    main()