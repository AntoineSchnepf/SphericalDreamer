import os
import torch
from torch.backends import cudnn
from torchvision import transforms
from types import SimpleNamespace
import tempfile
from typing import List
from PIL import Image

from submodules.EGformer.evaluate.evaluate import Evaluation
from submodules.EGformer.evaluate.data_loader import S3D_loader, Sample_loader
from submodules.EGformer.evaluate.pano_loader.pano_loader import Pano3D


class EGFormerDepthEngine:
    """
    Lazily initializes the EGFormer / Panoformer model once, and then
    reuses it for subsequent inferences.

    The public API is designed so that:
    - run_eval(...) has the same signature as before
    - get_egformer_depth(pil_list, **kwargs) also has the same signature
      as before (via a module-level singleton wrapper at the bottom).
    """

    def __init__(self):
        # --- lazy-init state ---
        self._initialized = False            # becomes True after first model init
        self._model_method = None            # "EGformer" or "Panoformer"
        self._align_type = None
        self._device = None                  # torch.device
        self._gpu_id = None                  # int
        self._config_base = None             # SimpleNamespace with base config
        self._transform = transforms.Compose([transforms.ToTensor()])

        # objects that are reused across calls
        self._evaluation = None              # Evaluation instance
        self._checkpoint_path = None

    # ------------------------------------------------------------------
    # low-level helpers (incorporate _main behaviour)
    # ------------------------------------------------------------------

    def _build_config(
        self,
        method: str = "EGformer",
        eval_data: str = "Inference",
        align_type: str = "Image",
        S3D_path: str = '',
        data_path: str = '',
        num_workers: int = 1,
        checkpoint_path: str = './checkpoints/EGFormer/pretrained_models/EGformer_pretrained.pkl',
        save_sample: bool = True,
        output_path: str = 'output',
        pano3d_root: str = '/YOUR_PATH/Pano3D_folder',
        pano3d_part: str = 'M3D_high',
        pano3d_split: str = './pano_loader/Pano3D/splits/M3D_v1_test.yaml',
        pano3d_types=None,
        world_size: int = 1,
        rank: int = 0,
        gpu: str = "0",
        dist_url: str = "tcp://127.0.0.1:7777",
        dist_backend: str = "nccl",
        multiprocessing_distributed: bool = True,
    ):
        """
        Builds a SimpleNamespace config exactly like your original run_eval().
        """
        if pano3d_types is None:
            pano3d_types = ['color', 'depth']

        config = SimpleNamespace(
            # high-level
            method=method,
            eval_data=eval_data,
            align_type=align_type,

            # paths
            S3D_path=S3D_path,
            data_path=data_path,

            num_workers=num_workers,
            checkpoint_path=checkpoint_path,

            save_sample=save_sample,
            output_path=output_path,

            # pano3d
            pano3d_root=pano3d_root,
            pano3d_part=pano3d_part,
            pano3d_split=pano3d_split,
            pano3d_types=pano3d_types,

            # ddp
            world_size=world_size,
            rank=rank,
            gpu=gpu,
            dist_url=dist_url,
            dist_backend=dist_backend,
            multiprocessing_distributed=multiprocessing_distributed,

            # will be set in _init_distributed_and_device()
            distributed=False,
        )
        return config

    def _init_distributed_and_device(self, config: SimpleNamespace):
        """
        Incorporates the CUDA + cudnn + GPU selection logic of _main/_worker,
        but adapted for a single-process, non-DDP setup.

        This avoids using torch.multiprocessing.spawn and avoids calling
        torch.distributed.init_process_group, so we don't hit the
        "Default process group has not been initialized" error.
        """
        # from _main
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
        cudnn.benchmark = True

        # We force single-process, non-distributed in this engine.
        # (If you want true DDP multi-process behaviour, use the original script.)
        config.world_size = 1
        config.rank = 0
        config.multiprocessing_distributed = False
        config.distributed = False  # <<< IMPORTANT: disables DDP in Evaluation

        # from _worker: choose primary GPU id
        self._gpu_id = int(str(config.gpu).split(',')[0])
        if torch.cuda.is_available():
            self._device = torch.device('cuda', self._gpu_id)
            print(f"Use GPU: {self._gpu_id} for evaluation")
        else:
            self._device = torch.device('cpu')
            print("CUDA not available, using CPU")

        return config

    def _build_dataset_and_loader(self, config: SimpleNamespace):
        """
        Incorporates the dataset selection part of _worker and creates
        a DataLoader. Uses self._transform instead of a local one.
        """
        # ----------------- dataset selection (from _worker) ----------------- #
        if config.eval_data == 'Structure3D':
            val_loader = S3D_loader(
                config.S3D_path,
                transform=self._transform,
                transform_t=self._transform,
            )

        elif config.eval_data == 'Pano3D':
            val_loader = Pano3D(
                root=config.pano3d_root,
                part=config.pano3d_part,
                split=config.pano3d_split,
                types=config.pano3d_types,
            )

        elif config.eval_data == 'Inference':
            val_loader = Sample_loader(
                config.data_path,
                transform=self._transform,
                transform_t=self._transform,
            )

        else:
            raise ValueError("Check the command option (eval_data)")

        # from _worker: build DataLoader
        dataloader = torch.utils.data.DataLoader(
            val_loader,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        return dataloader

    # ------------------------------------------------------------------
    # lazy model init
    # ------------------------------------------------------------------

    def _init_model_if_needed(self, config: SimpleNamespace):
        """
        This is the key lazy-init function:
        - On the first call, builds the Evaluation object (which in turn
          will build the EGFormer/Panoformer model the first time you
          call evaluate_egformer/evaluate_panoformer).
        - On later calls, we reuse self._evaluation and only swap
          val_dataloader.
        """
        if self._initialized:
            return

        # Remember some "base" knobs so we can reuse them
        self._model_method = config.method
        self._align_type = config.align_type
        self._checkpoint_path = config.checkpoint_path

        # Initialize CUDA / non-distributed behaviour once
        config = self._init_distributed_and_device(config)

        # Build a dummy loader for initialization (we don't care about the data yet)
        dummy_loader = self._build_dataset_and_loader(config)

        # This mirrors the Evaluation(...) call in _worker, but we keep
        # the instance to reuse later.
        self._evaluation = Evaluation(
            config,
            dummy_loader,
            self._gpu_id,
        )

        # Store base config for later cloning
        self._config_base = config
        self._initialized = True

    # ------------------------------------------------------------------
    # public: run_eval (config-level) + PIL helper
    # ------------------------------------------------------------------

    def run_eval(self, **run_eval_kwargs):
        """
        Drop-in replacement for your original run_eval(...), but using
        lazy model initialization and reusing the same Evaluation object.

        For each call:
        - ensure model is initialized
        - clone base config but override data_path/output_path/etc.
        - rebuild DataLoader
        - swap self._evaluation.val_dataloader
        - run evaluate_egformer / evaluate_panoformer
        """
        # Build config from kwargs, with the same defaults as before
        config = self._build_config(**run_eval_kwargs)

        # Ensure model / CUDA are initialized only once
        self._init_model_if_needed(config)

        # Clone base config and override a few fields for this specific run
        config_eval = SimpleNamespace(**vars(self._config_base))
        config_eval.data_path = config.data_path
        config_eval.output_path = config.output_path
        config_eval.eval_data = config.eval_data
        config_eval.method = config.method
        config_eval.align_type = config.align_type
        config_eval.save_sample = config.save_sample

        # Build the dataloader for this *actual* data_path
        val_dataloader = self._build_dataset_and_loader(config_eval)
        self._evaluation.val_dataloader = val_dataloader

        # This mirrors the bottom of _worker
        if config_eval.method == "EGformer":
            self._evaluation.evaluate_egformer()
        elif config_eval.method == "Panoformer":
            self._evaluation.evaluate_panoformer()
        else:
            raise ValueError("Check Command options (method)")

    def get_egformer_depth(self, pil_list: List[Image.Image], **run_eval_kwargs):
        """
        Same high-level behaviour as your original get_egformer_depth(...),
        but going through the lazy-initialized engine.

        - Writes PIL inputs into a temporary directory
        - Calls run_eval() so EGformer writes its outputs
        - Reads all PNG/JPEG outputs back into a list of PIL images.
        """

        # Sensible defaults; user can override them via **run_eval_kwargs
        default_kwargs = dict(
            method="EGformer",
            eval_data="Inference",        # force Sample_loader path
            multiprocessing_distributed=False,  # no DDP in this path
            world_size=1,
            gpu="0",
            save_sample=True,             # ensure outputs are written
        )
        default_kwargs.update(run_eval_kwargs)

        with tempfile.TemporaryDirectory() as input_root, tempfile.TemporaryDirectory() as output_root:
            # 1. Create a subdirectory as Sample_loader expects
            sample_dir = os.path.join(input_root, "sample_0")
            os.makedirs(sample_dir, exist_ok=True)

            # 2. Save input PIL images inside that subdirectory
            for i, im in enumerate(pil_list):
                in_path = os.path.join(sample_dir, f"img_{i:04d}.png")
                im.save(in_path, format="PNG")

            # 3. Run evaluation on the parent directory
            self.run_eval(
                data_path=input_root,
                output_path=output_root,
                **default_kwargs,
            )

            # 4. Read back all PNG/JPEG outputs from the output_root
            output_images = []
            for root, _, files in os.walk(output_root):
                for fname in sorted(files):
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    out_path = os.path.join(root, fname)
                    img = Image.open(out_path).convert("RGB")
                    output_images.append(img)

        return output_images


# ----------------------------------------------------------------------
# Module-level "singleton" and top-level functions to keep your API
# ----------------------------------------------------------------------

_EGFORMER_ENGINE = EGFormerDepthEngine()


def run_eval(
        method: str = "EGformer",
        eval_data: str = "Inference",
        align_type: str = "Image",
        S3D_path: str = '',
        data_path: str = '',
        num_workers: int = 1,
        checkpoint_path: str = './checkpoints/EGFormer/pretrained_models/EGformer_pretrained.pkl',
        save_sample: bool = True,
        output_path: str = 'output',
        pano3d_root: str = '/YOUR_PATH/Pano3D_folder',
        pano3d_part: str = 'M3D_high',
        pano3d_split: str = './pano_loader/Pano3D/splits/M3D_v1_test.yaml',
        pano3d_types=None,
        world_size: int = 1,
        rank: int = 0,
        gpu: str = "0",
        dist_url: str = "tcp://127.0.0.1:7777",
        dist_backend: str = "nccl",
        multiprocessing_distributed: bool = True,
    ):
    """
    Wrapper that forwards to the singleton engine, preserving the original
    run_eval(...) signature and default values.
    """
    return _EGFORMER_ENGINE.run_eval(
        method=method,
        eval_data=eval_data,
        align_type=align_type,
        S3D_path=S3D_path,
        data_path=data_path,
        num_workers=num_workers,
        checkpoint_path=checkpoint_path,
        save_sample=save_sample,
        output_path=output_path,
        pano3d_root=pano3d_root,
        pano3d_part=pano3d_part,
        pano3d_split=pano3d_split,
        pano3d_types=pano3d_types,
        world_size=world_size,
        rank=rank,
        gpu=gpu,
        dist_url=dist_url,
        dist_backend=dist_backend,
        multiprocessing_distributed=multiprocessing_distributed,
    )


def get_egformer_depth(pil_list: List[Image.Image], **run_eval_kwargs):
    """
    Wrapper that forwards to the singleton engine, preserving your original
    get_egformer_depth(...) public API.
    """
    return _EGFORMER_ENGINE.get_egformer_depth(pil_list, **run_eval_kwargs)

if __name__ == "__main__":
    inputs = [
        Image.open("/home/a.schnepf/phd/SphericalDreamer/submodules/EGformer/evaluate/INFER_SAMPLE/test_sample/S3D_sample.png"),
        Image.open("/home/a.schnepf/phd/SphericalDreamer/submodules/EGformer/evaluate/INFER_SAMPLE/test_sample/test_sample.png"),
    ]
    outputs = get_egformer_depth(inputs)