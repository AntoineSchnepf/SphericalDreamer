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
from harmonic_blending import harmonic_blend_of_depths
import my_utils
from my_utils import printc
import ldi_inpaiting as ldi

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

phase2a_output_prefix = "02a_"
output_prefix = "02b_"

if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=False, 
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
    # ------------ PHASE 2-B. BACKGROUND RGBD INPAINTING -------------- #
    # ----------------------------------------------------------------- #
    printc(f"=== [PHASE 2-B] EXPERIMENT: {config.expname} ===", color='cyan')
    # if not config.load_phase1_from:
    if config.phase2.apply_ldi:
        if True: # TODO: @Karim. Disabling skipping for now. Can you fix it later ?
            printc("=== PHASE 2-B: BACKGROUND RGBD INPAINTING ===", color='green')

            # -----------------------------
            # 0. LOAD INPUT IMAGES + DEPTH
            # -----------------------------
            list_img = []
            list_depth_origin = []
            for i in range(1, config.num_dreams):
                printc(f"--- 2-B: load image  {i:02d} / {config.num_dreams} ---", color='yellow')

                save_dir__ = os.path.join(save_dir_, f"align_{i:02d}")

                data =  np.load(f"{save_dir__}/{phase2a_output_prefix}YY_other.npy", allow_pickle=True).item()
                
                depth_estimated       = data['depth_estimated']
                # pose_intermediate     = data['pose_intermediate']
                # warped_img_interp     = data['warped_img_interp']
                # warped_depth_interp   = data['warped_depth_interp']
                pano_rgb_inpainted    = data['pano_rgb_inpainted']
                # missing_info_mask     = data['missing_info_mask']

                list_img.append(my_utils.PIL_to_numpy(pano_rgb_inpainted))
                list_depth_origin.append(depth_estimated)

            # ---------------------------------------
            # I. COMPUTE MASK FOR FOREGROUND OBJECTS
            # ---------------------------------------
            t0 = time.time()
            list_mask = []
            sam, mask_generator = ldi.instanciate_sam(config)

            for i in range(1, config.num_dreams):
                printc(f"--- 2-B: Compute mask for foreground object  {i:02d} / {config.num_dreams} ---", color='yellow')
                mask = ldi.get_foreground_segmask(
                    config,
                    mask_generator, 
                    list_img[i-1],
                    list_depth_origin[i-1],
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
            llm_model, processor = ldi.instanciate_llm_and_processor()

            for i in range(1, config.num_dreams):
                printc(f"--- 2-B: Lama Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
                prompt, mask_smooth_pil, inpaint_pano_lama_pil, viz_kwargs = ldi.lama_flux_double_inpainting_p1(
                    config,
                    spherical_dreamer,
                    llm_model,
                    processor,
                    image=list_img[i-1],
                    mask=list_mask[i-1],
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

            for i in range(1, config.num_dreams):
                printc(f"--- 2-B: Flux Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
                inpaint_pano_pil, inpaint_mask_pil = ldi.lama_flux_double_inpainting_p2(
                    config,
                    spherical_dreamer,
                    list_prompt[i-1],
                    list_mask_smooth_pil[i-1],
                    list_inpaint_pano_lama_pil[i-1],
                    list_viz_kwargs[i-1],
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

            # preparation
            if config.phase_ldi.depth_inpainting.method == "infusion":
                pipe_dp = ldi.instanciate_pipe_dp()

            for i in range(1, config.num_dreams):
                printc(f"--- 2-B: Depth Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')

                img_pil, depth_origin, inpaint_mask_pil_, inpaint_mask_bool_ = ldi.prepare_inpainting(
                    config,
                    list_img[i-1],
                    list_depth_origin[i-1],
                    list_inpaint_mask_pil[i-1],
                )

                if config.phase_ldi.depth_inpainting.method == "harmonic_blending":
                        depth_360_mono = spherical_dreamer.estimate_pano_depth(inpaint_pano_pil)
                        inpaint_pano = np.array(inpaint_pano_pil) / 255.0
                        _, _, _, depth_inpainted_hblending = harmonic_blend_of_depths(
                            colors=inpaint_pano, 
                            warped_depth_interp=depth_origin, #gt depth
                            depth_estimated=depth_360_mono, # new depth
                            missing_info_mask=inpaint_mask_bool_,
                            pose= np.eye(4).astype(np.float32),
                            sphere_radius=1.0,
                            height=inpaint_pano.shape[0],
                            width=inpaint_pano.shape[1],
                            logging=False, 
                        )

                elif config.phase_ldi.depth_inpainting.method == "infusion":
                    depth_inpainted = ldi.inpaint_bg_depth(
                        image=img_pil,
                        depth=depth_origin,
                        image_bg=list_inpaint_pano_pil[i-1],
                        bg_mask=inpaint_mask_pil,
                        pipe_dp=pipe_dp,
                        rescale_to_min_depth=True,
                        plot_results=False,
                        pad_width=config.phase_ldi.depth_inpainting.pad_width,
                    )

                elif config.phase_ldi.depth_inpainting.method == "nearest":
                    depth_inpainted = ldi.interpolate_depth_nearest(
                        depth=depth_origin,
                        bg_mask=inpaint_mask_bool_,
                        pad_width=config.phase_ldi.depth_inpainting.pad_width,
                    )
                
                elif config.phase_ldi.depth_inpainting.method == "bilinear_plus_nn":
                    depth_inpainted = ldi.interpolate_depth_bilinear_plus_nn(
                        depth=depth_origin,
                        bg_mask=inpaint_mask_bool_,
                        pad_width=config.phase_ldi.depth_inpainting.pad_width,
                    )
                
                else:
                    raise ValueError(f"Unknown depth inpainting method: {config.phase_ldi.depth_inpainting.method}")

                depth_inpainted[~inpaint_mask_bool_] = np.nan
                if config.phase_ldi.depth_inpainting.apply_post_processing:
                    depth_inpainted = ldi.post_process_inpainted_depth(
                        depth_bg=depth_inpainted,
                        depth_fg=depth_origin,
                        bg_mask=inpaint_mask_bool_,
                    )
                list_depth_inpainted.append(depth_inpainted)

            if config.phase_ldi.depth_inpainting.method == "infusion":
                del pipe_dp
            torch.cuda.empty_cache()
            print(f"Depth inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images with method {config.phase_ldi.depth_inpainting.method}.")

            # SAVE RESULTS
            for i in range(1, config.num_dreams):
                my_utils.save_rgbd_ldi_pano(
                    pano_rgb_bg=list_inpaint_pano_pil[i-1],
                    depth_bg=list_depth_inpainted[i-1],
                    mask_bg=my_utils.pil_mask_to_numpy_bool(list_inpaint_mask_pil[i-1]),
                    dream=i,
                    save_dir_=save_dir_,
                    phase=2
                ) 
            printc("PHASE 1-B SUCCESSFULLY COMPLETED!", color='green')
        else:
            printc("SKIPPING PHASE 1-B: BACKGROUND RGBD INPAINTING", color='magenta')
            # TODO: Karim should implement his skipping strategy here.
    else:
        printc("PHASE 1-B: LDI INPAINTING NOT APPLIED AS PER CONFIGURATION", color='magenta')