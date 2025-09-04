import os
import sys 
import cv2
import numpy as np
import argparse
import torch.nn.functional as F
import logging
from PIL import Image

_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 

from utility import depthmap_utils, pointcloud_utils

from utils.depth_alignment import Pano_depth_estimation

# deactivagte logging
logging.disable(logging.CRITICAL + 1)


class Gen_panodepth:
    def __init__(self, depth_model='DepthAnythingv2', save_dir='outputs_panodepth', pathtxt=None , device='cuda'):
        self.save_dir = save_dir
        self.device = device
        self.depth_model = depth_model
        self.pathtxt = pathtxt
        os.makedirs(self.save_dir, exist_ok=True)

    def read_paths(self, file_path):
        rgbpath_list = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    path = line.strip() 
                    if path:  
                        rgbpath_list.append(path)
        except FileNotFoundError:
            print(f"file {file_path} not found")
        except Exception as e:
            print(f"error in reading: {e}")
        
        return rgbpath_list

    def readImg(self, path):
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        return img

    def generate_panodepth(self, pano_rgb):
        """
        args:
            `pano_rgb`: np.array of shape [pano_h,pano_w,3] and values in [0-255]        
        """
        pano_h, pano_w = pano_rgb.shape[0], pano_rgb.shape[1]
        panodepth_estimator = Pano_depth_estimation(pano_h, pano_w, self.save_dir, self.device, depth_model=self.depth_model)
        pano_depth = panodepth_estimator.get_panodepth(pano_rgb)  #[0-1] 

        return pano_depth  

    def run(self, image, save_dir=None):

        depth = self.generate_panodepth(image)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            depth_path = f'{save_dir}/depth.png'
            np.save(f'{save_dir}/depth.npy', depth)
            depthmap_utils.depth_visual_save(depth, f'{save_dir}/depth_rgb.png')
            pcd = pointcloud_utils.depthmap2pointcloud_erp(depth , image, f"{save_dir}/pcd_rgb.ply", return_pcd=True)

            # save depth as PIL image
            depth_pil = (depth -  depth.min()) / (depth.max() - depth.min())
            max_val = 65535
            depth_pil = (depth_pil * max_val).astype(np.uint16)
            depth_pil = Image.fromarray(depth_pil)
            depth_pil.save(depth_path)

            return depth, pcd

        return depth


depth_model="DepthAnythingv2"
# depth_model = "zoedepth"
save_dir="OUTPUTS/gen_panodepth"
input_path='OUTPUTS/dream_explore/inpainted_image.png'

gen_panodepth = Gen_panodepth(depth_model = depth_model,save_dir=save_dir)
image = gen_panodepth.readImg(input_path)
depth, pcd = gen_panodepth.run(image, save_dir) 