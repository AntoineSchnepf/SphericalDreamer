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

_phase_1a = "1a"
_phase_1b = "1b"
_phase_2a = "2a"
_phase_2b = "2b"
_phase_2c = "2c"

_phase_current = _phase_1b

def get_save_viz_path(dream):
    return save_dir_ / f"dream_{dream:02d}" / _phase_current / "ldi_insights"

if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "Antoine/debug.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)
    plot_results = config.ldi.save_plots

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    # ----------------------------------------------------- #
    # ------------ PHASE 1-B. LDI INPAINTING -------------- #
    # ----------------------------------------------------- #
    printc(f"=== [PHASE {_phase_current}] EXPERIMENT: {config.expname} ===", color='cyan')
    if not config.load_phase1b_from:
        printc(f"=== PHASE {_phase_current}: LDI INPAINTING ===", color='green')

        # -----------------------------
        # 0. LOAD INPUT IMAGES + DEPTH
        # -----------------------------
        list_img = []
        list_depth_origin = []
        for i in range(config.num_dreams):
            printc(f"--- {_phase_current}: load image  {i:02d} / {config.num_dreams} ---", color='yellow')

            img, depth_origin = my_utils.load_rgbd_pano(
                dream=i,
                save_dir_=save_dir_,
                phase=_phase_1a,
            )
            list_img.append(img)
            list_depth_origin.append(depth_origin)

        # ---------------------------------------
        # 1. COMPUTE MASK FOR FOREGROUND OBJECTS
        # ---------------------------------------
        t0 = time.time()
        list_mask = []
        sam, mask_generator = ldi.instanciate_sam(config)

        for i in range(config.num_dreams):
            printc(f"--- {_phase_current}: Compute mask for foreground object  {i:02d} / {config.num_dreams} ---", color='yellow')
            
            save_viz_path = get_save_viz_path(i)
            os.makedirs(save_viz_path, exist_ok=True)

            mask = ldi.get_foreground_segmask(
                config,
                mask_generator, 
                list_img[i],
                list_depth_origin[i],
                plot_results=plot_results,
                save_path=save_viz_path,
            )
            list_mask.append(mask)

        del sam
        del mask_generator
        torch.cuda.empty_cache()
        print(f"Foreground mask computed in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

        # ------------------------
        # 2. INPAINTING WITH LAMA
        # ------------------------
        t0 = time.time()
        list_prompt=[]
        list_mask_smooth_pil = []
        list_inpaint_pano_lama_pil = []
        list_viz_kwargs = []
        llm_model, processor = ldi.instanciate_llm_and_processor()

        for i in range(config.num_dreams):
            printc(f"--- {_phase_current}: Lama Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
            prompt, mask_smooth_pil, inpaint_pano_lama_pil, viz_kwargs = ldi.lama_flux_double_inpainting_p1(
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
        # 3. INPAINTING WITH FLUX
        # --------------------------

        t0 = time.time()
        list_inpaint_mask_pil = []
        list_inpaint_pano_pil = []

        for i in range(config.num_dreams):
            printc(f"--- {_phase_current}: Flux Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
            inpaint_pano_pil, inpaint_mask_pil = ldi.lama_flux_double_inpainting_p2(
                config,
                spherical_dreamer,
                list_prompt[i],
                list_mask_smooth_pil[i],
                list_inpaint_pano_lama_pil[i],
                list_viz_kwargs[i],
                plot_results=plot_results,
                save_path=get_save_viz_path(i),
            )
            list_inpaint_mask_pil.append(inpaint_mask_pil)
            list_inpaint_pano_pil.append(inpaint_pano_pil)

        spherical_dreamer._release_flux_inpainting_memory()
        torch.cuda.empty_cache()
        print(f"FLUX inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

        # -------------------------------------------------
        # 4. DEPTH INPAINTING (at resolution 1024 * 2048)
        # -------------------------------------------------
        t0 = time.time()
        list_depth_inpainted = []
        list_mask_inpaint_resized = []
        list_depth_origin_resized = []
        list_img_pil_resized = []

        # preparation
        if config.ldi.depth_inpainting.method == "infusion":
            pipe_dp = ldi.instanciate_pipe_dp()

        for i in range(config.num_dreams):
            printc(f"--- {_phase_current}: Depth Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
            
            img_pil, depth_origin, _, inpaint_mask_bool_ = ldi.prepare_inpainting(
                config,
                list_img[i],
                list_depth_origin[i],
                list_inpaint_mask_pil[i],
            )

            if config.ldi.depth_inpainting.method == "horizontal_min_prior":
                _, depth_prior = ldi.remove_low_freq(depth_origin, config=config.ldi.masking.depth_mean_based.remove_depth_low_freq)
                inpaint_mask_bool_ = np.ones_like(inpaint_mask_bool_, dtype=bool)  # in this method, we inpaint everywhere
                _ = my_utils.numpy_bool_to_pil_mask(inpaint_mask_bool_)
                depth_inpainted = depth_prior

            elif config.ldi.depth_inpainting.method == "harmonic_blending":
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
                        phase=_phase_current,
                        logging=plot_results,
                        where_save=save_viz_path,
                    )

            elif config.ldi.depth_inpainting.method == "infusion":
                depth_inpainted = ldi.inpaint_bg_depth_infusion(
                    image=img_pil,
                    depth=depth_origin,
                    image_bg=list_inpaint_pano_pil[i],
                    bg_mask=inpaint_mask_pil,
                    pipe_dp=pipe_dp,
                    rescale_to_min_depth=True,
                    pad_width=config.ldi.depth_inpainting.pad_width,
                    plot_results=plot_results,
                    save_path=save_viz_path,
                )

            elif config.ldi.depth_inpainting.method == "nearest":
                depth_inpainted = ldi.interpolate_depth_nearest(
                    depth=depth_origin,
                    bg_mask=inpaint_mask_bool_,
                    pad_width=config.ldi.depth_inpainting.pad_width,
                )
            
            elif config.ldi.depth_inpainting.method == "bilinear_plus_nn":
                depth_inpainted = ldi.interpolate_depth_bilinear_plus_nn(
                    depth=depth_origin,
                    bg_mask=inpaint_mask_bool_,
                    pad_width=config.ldi.depth_inpainting.pad_width,
                )
            
            else:
                raise ValueError(f"Unknown depth inpainting method: {config.ldi.depth_inpainting.method}")


            depth_inpainted[~inpaint_mask_bool_] = np.nan
            if config.ldi.depth_inpainting.apply_post_processing:
                depth_inpainted = ldi.post_process_inpainted_depth(
                    depth_bg=depth_inpainted,
                    depth_fg=depth_origin,
                    bg_mask=inpaint_mask_bool_,
                    plot=plot_results,
                    save_path=get_save_viz_path(i),
                )
                
            list_depth_inpainted.append(depth_inpainted)
            list_mask_inpaint_resized.append(inpaint_mask_bool_)
            list_depth_origin_resized.append(depth_origin)
            list_img_pil_resized.append(img_pil)

        if config.ldi.depth_inpainting.method == "infusion":
            del pipe_dp
        torch.cuda.empty_cache()
        print(f"Depth inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images with method {config.ldi.depth_inpainting.method}.")

        # SAVE RESULTS
        for i in range(config.num_dreams):
            if config.ldi.depth_inpainting.method == "horizontal_min_prior":
                suffix="hminprior"
            elif config.ldi.depth_inpainting.method == "harmonic_blending":
                suffix="hblending"
            elif config.ldi.depth_inpainting.method == "infusion":
                suffix="infusion"
            elif config.ldi.depth_inpainting.method == "nearest":
                suffix="nn"
            elif config.ldi.depth_inpainting.method == "bilinear_plus_nn":
                suffix="bilinear_nn"

            kwargs = {
                "img_pil": list_img_pil_resized[i],
                "inpaint_pano_pil": list_inpaint_pano_pil[i],
                "inpaint_mask_pil": list_mask_inpaint_resized[i],
                "depth_origin": list_depth_origin_resized[i],
                f"depth_inpainted_{suffix}": list_depth_inpainted[i]
            }
            ldi.visualize_depth_inpainting(
                **kwargs,
                save_path= get_save_viz_path(i) / "07_depth_inpainting_visualization.png",
            )
            my_utils.save_rgbd_ldi_pano(
                pano_rgb_bg=list_inpaint_pano_pil[i],
                depth_bg=list_depth_inpainted[i],
                mask_bg=list_mask_inpaint_resized[i],
                dream=i,
                save_dir_=save_dir_,
                phase=_phase_current,
            ) 
        printc(f"PHASE {_phase_current} SUCCESSFULLY COMPLETED!", color='green')
    else:
        printc(f"SKIPPING PHASE {_phase_current}: LDI INPAINTING", color='magenta')


        source_phase1b_path = Path(config.save_dir) / config.load_phase1b_from
        dest_phase1b_path = Path(save_dir_)

        my_utils.copy_phase_folders(
            source_dir=source_phase1b_path,
            dest_dir=dest_phase1b_path,
            phase=_phase_current,
        )