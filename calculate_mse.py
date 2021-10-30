import argparse
import os
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataSet

from torch.nn import MSELoss

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
    assert len(dataset1) == len(dataset2)
    n_total_data = len(dataset1)

    batch_size = args.batch_size
    dataloader1 = DataLoader(dataset1 , batch_size=batch_size, shuffle=False,  num_workers=4, drop_last=False)
    dataloader2 = DataLoader(dataset2 , batch_size=batch_size, shuffle=False,  num_workers=4, drop_last=False)
    

    mse_function = MSELoss(reduction='sum') # mini-batch processing 
    sum = 0
    for imgs1,imgs2 in tqdm(zip(dataloader1,dataloader2)):
        if args.device == 'cuda': 
            imgs1 = imgs1.cuda()
            imgs2 = imgs2.cuda()
        with torch.no_grad():
            sum += mse_function(imgs1.flatten(start_dim=1), imgs2.flatten(start_dim=1)).item()/(imgs1.flatten(start_dim=1).shape[1])  # also devide by # C*H*W
    mse_score = sum/n_total_data
    print(f"total avg mse score: {mse_score}")

    with open(args.output_log,'w') as f:
        f.write(f"total avg mse score: {mse_score}")
    print(f"result has been saved to {args.output_log}")

    print("the process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d1','--dataset_path1', type=str, default='/home/nessessence/DDPM/stylegan2-pytorch/datasets/CelebA-HQ-img')
    parser.add_argument('-d2','--dataset_path2', type=str, default='/data/nessessence/DDPM/projected_output/W/inversed_imgs/')
    parser.add_argument('-gpu','--gpu_ids', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda','cpu'])
    parser.add_argument('-s','--img_size', type=int, default=128)
    parser.add_argument('-b','--batch_size', type=int, default=4096)
    parser.add_argument('-o','--output_log', type=str, default='output/w_mse_result.txt')
    main()