import os
import warnings
import logging
import contextlib
from io import StringIO
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import my_utils
from my_utils import printc
import rgbd_foreground_inpaiting as rgbd_inpaint

# Disabling some warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["GLOG_logtostderr"] = "0"
os.environ["CERES_MINIMIZER_PROGRESS_TO_STDOUT"] = "0"
logging.disable(logging.CRITICAL + 1)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", FutureWarning)

# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer


if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=True, 
        debug_parser_override=["--config", "Antoine/F0_forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)
    plot_results = config.phase_ldi.interactive_plot_results

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    # ----------------------------------------------------------------- #
    # ------------ PHASE 1-B. BACKGROUND RGBD INPAINTING -------------- #
    # ----------------------------------------------------------------- #
    printc(f"=== [PHASE 1-B] EXPERIMENT: {config.expname} ===", color='cyan')
    if not config.load_phase1_from:
        printc("=== PHASE 1-B: BACKGROUND RGBD INPAINTING ===", color='green')

        # -----------------------------
        # 0. LOAD INPUT IMAGES + DEPTH
        # -----------------------------
        list_img = []
        list_depth_origin = []
        for i in range(config.num_dreams):
            printc(f"--- 1-B: load image  {i:02d} / {config.num_dreams} ---", color='yellow')

            img, depth_origin = my_utils.load_rgbd_pano(
                dream=i,
                save_dir_=save_dir_
            )
            list_img.append(img)
            list_depth_origin.append(depth_origin)

        # -----------------------------------------
        # I. COMPUTE MASK FOR FOREGROUND OBJECTS
        # -----------------------------------------
        t0 = time.time()
        list_mask = []
        sam, mask_generator = rgbd_inpaint.instanciate_sam(config)

        for i in range(config.num_dreams):
            printc(f"--- 1-B: Compute mask for foreground object  {i:02d} / {config.num_dreams} ---", color='yellow')
            mask = rgbd_inpaint.get_foreground_segmask(
                config,
                mask_generator, 
                list_img[i],
                list_depth_origin[i],
                plot_results=plot_results,
            )
            list_mask.append(mask)

        del sam
        del mask_generator
        torch.cuda.empty_cache()
        print(f"Foreground mask computed in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

        # ------------------------
        # II. INPAINTING WITH LAMA
        # ------------------------
        t0 = time.time()
        list_prompt=[]
        list_mask_smooth_pil = []
        list_inpaint_pano_lama_pil = []
        list_viz_kwargs = []
        llm_model, processor = rgbd_inpaint.instanciate_llm_and_processor()

        for i in range(config.num_dreams):
            printc(f"--- 1-B: Lama Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
            prompt, mask_smooth_pil, inpaint_pano_lama_pil, viz_kwargs = rgbd_inpaint.lama_flux_double_inpainting_p1(
                config,
                spherical_dreamer,
                llm_model,
                processor,
                image=list_img[i],
                mask=list_mask[i],
            )
            list_prompt.append(prompt)
            list_mask_smooth_pil.append(mask_smooth_pil)
            list_inpaint_pano_lama_pil.append(inpaint_pano_lama_pil)
            list_viz_kwargs.append(viz_kwargs)

        spherical_dreamer._release_lama_memory()
        del llm_model
        del processor
        torch.cuda.empty_cache()
        print(f"Lama inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

        # --------------------------
        # III. INPAINTING WITH FLUX
        # --------------------------

        t0 = time.time()
        list_inpaint_mask_pil = []
        list_inpaint_pano_pil = []

        for i in range(config.num_dreams):
            printc(f"--- 1-B: Flux Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
            inpaint_pano_pil, inpaint_mask_pil = rgbd_inpaint.lama_flux_double_inpainting_p2(
                config,
                spherical_dreamer,
                list_prompt[i],
                list_mask_smooth_pil[i],
                list_inpaint_pano_lama_pil[i],
                list_viz_kwargs[i],
                plot_results=plot_results,
            )
            list_inpaint_mask_pil.append(inpaint_mask_pil)
            list_inpaint_pano_pil.append(inpaint_pano_pil)

        spherical_dreamer._release_flux_inpainting_memory()
        torch.cuda.empty_cache()
        print(f"FLUX inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

        # -------------------------------------------------
        # III. DEPTH INPAINTING (at resolution 1024 * 2048)
        # -------------------------------------------------
        t0 = time.time()
        list_depth_inpainted = []
        pipe_dp = rgbd_inpaint.instanciate_pipe_dp()

        for i in range(config.num_dreams):
            printc(f"--- 1-B: Depth Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')

            he = config.phase_ldi.inpainting.flux_inpainting_resolution.height
            wi = config.phase_ldi.inpainting.flux_inpainting_resolution.width

            img_pil = my_utils.numpy_to_PIL(my_utils.opencv_resize(list_img[i], he, wi))
            depth_origin = my_utils.opencv_resize(list_depth_origin[i], he, wi) # FLAG: depth resize
            inpaint_mask_pil = list_inpaint_mask_pil[i].resize((wi, he), resample=Image.NEAREST)
            depth_inpainted = rgbd_inpaint.inpaint_bg_depth(
                image=img_pil,
                depth=depth_origin,
                image_bg=list_inpaint_pano_pil[i],
                bg_mask=inpaint_mask_pil,
                pipe_dp=pipe_dp,
                rescale_to_min_depth=True,
                plot_results=plot_results,
            )
            list_depth_inpainted.append(depth_inpainted)

        del pipe_dp
        torch.cuda.empty_cache()
        print(f"Depth inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

        # SAVE RESULTS
        for i in range(config.num_dreams):
            my_utils.save_rgbd_ldi_pano(
                pano_rgb_bg=list_inpaint_pano_pil[i],
                depth_bg=list_depth_inpainted[i],
                mask_bg=my_utils.pil_mask_to_numpy_bool(list_inpaint_mask_pil[i]),
                dream=i,
                save_dir_=save_dir_,
                phase=1
            ) 
        printc("PHASE 1-B SUCCESSFULLY COMPLETED!", color='green')
    else:
        printc("SKIPPING PHASE 1-B: BACKGROUND RGBD INPAINTING", color='magenta')
        # TODO: Karim should implement its skipping strategy here.