import os
import torch
from torch.backends import cudnn
from torchvision import transforms
from types import SimpleNamespace
import os
import tempfile
from typing import List
from PIL import Image
import os
import tempfile
from typing import List

from PIL import Image

from submodules.EGformer.evaluate.evaluate import Evaluation
from submodules.EGformer.evaluate.data_loader import S3D_loader, Sample_loader
from submodules.EGformer.evaluate.pano_loader.pano_loader import Pano3D


def _main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    cudnn.benchmark = True

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size
        torch.multiprocessing.spawn(
            _worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, config),
        )
    else:
        # if gpu is a string like "0,1,2,3", we pass it as-is (same as your original)
        _worker(config.gpu, ngpus_per_node, config)

def _worker(gpu, ngpus_per_node, config):
    transform = transforms.Compose([transforms.ToTensor()])

    # if gpu is like "0" or 0, this is fine
    device_ids = [int(str(config.gpu).split(',')[0])]

    if config.gpu is not None:
        print(f'Use GPU: {gpu} for evaluation')

    if config.distributed:
        if config.dist_url == "envs://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.rank,
        )

    # ----------------- dataset selection ----------------- #
    if config.eval_data == 'Structure3D':
        val_loader = S3D_loader(
            config.S3D_path,
            transform=transform,
            transform_t=transform,
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
            transform=transform,
            transform_t=transform,
        )

    else:
        print("Check the command option (eval_data)")
        return

    device = torch.device('cuda', device_ids[0])

    val_dataloader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    evaluation = Evaluation(
        config,
        val_dataloader,
        gpu,
    )

    if config.method == "EGformer":
        evaluation.evaluate_egformer()
    elif config.method == "Panoformer":
        evaluation.evaluate_panoformer()
    else:
        print("Check Command options (method)")

def run_eval(
        # ---- original high-level options ----
        method: str = "EGformer",                 # ["EGformer", "Panoformer"]
        eval_data: str = "Inference",           # ["Structure3D", "Pano3D", "Inference"]
        align_type: str = "Image",                # ["Column", "Image"]

        # ---- data paths ----
        S3D_path: str = '',                       # Structure3D data path
        data_path: str = '',                      # inference data path

        num_workers: int = 1,

        checkpoint_path: str = './checkpoints/EGFormer/pretrained_models/EGformer_pretrained.pkl',

        save_sample: bool = True,
        output_path: str = 'output',

        # ---- Pano3D options ----
        pano3d_root: str = '/YOUR_PATH/Pano3D_folder',
        pano3d_part: str = 'M3D_high',
        pano3d_split: str = './pano_loader/Pano3D/splits/M3D_v1_test.yaml',
        pano3d_types=None,  # default ["color", "depth"]

        # ---- DDP options ----
        world_size: int = 1,
        rank: int = 0,
        gpu: str = "0",
        dist_url: str = "tcp://127.0.0.1:7777",
        dist_backend: str = "nccl",
        multiprocessing_distributed: bool = True,
    ):
    """
    Run evaluation with defaults matching the old argparse-based script.

    Example usage:

        run_eval(
            method="EGformer",
            eval_data="Inference",
            data_path="/path/to/images",
            checkpoint_path="checkpoints/EGFormer/pretrained_models/EGformer_pretrained.pkl",
            output_path="OUTPUTS/EGFormer",
            gpu="0",
            world_size=1,
            multiprocessing_distributed=False,
        )
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

        # will be set in _main()
        distributed=False,
    )

    _main(config)

def get_egformer_depth(pil_list: List[Image.Image], **run_eval_kwargs):
    """
    Takes a list of PIL images, writes them into a temporary folder with the
    directory structure expected by Sample_loader, runs run_eval(), and
    returns the output images as a list of PIL.Image objects.
    """

    # Sensible defaults; user can override them via **run_eval_kwargs
    default_kwargs = dict(
        method="EGformer",
        eval_data="Inference",        # force Sample_loader path
        multiprocessing_distributed=True,
        world_size=1,
        gpu="0",
        save_sample=True,             # ensure outputs are written
    )
    default_kwargs.update(run_eval_kwargs)

    # Import your existing run_eval (adjust import path if needed)

    with tempfile.TemporaryDirectory() as input_root, tempfile.TemporaryDirectory() as output_root:
        # 1. Create a subdirectory as Sample_loader expects
        sample_dir = os.path.join(input_root, "sample_0")
        os.makedirs(sample_dir, exist_ok=True)

        # 2. Save input PIL images inside that subdirectory
        for i, im in enumerate(pil_list):
            in_path = os.path.join(sample_dir, f"img_{i:04d}.png")
            im.save(in_path, format="PNG")

        # 3. Run evaluation on the parent directory
        run_eval(
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



if __name__ == "__main__":
    # run_eval(
    #     data_path="/home/a.schnepf/phd/SphericalDreamer/submodules/EGformer/evaluate/INFER_SAMPLE",
    #     output_path="/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/EGFormer",
    # )

    inputs = [
        Image.open("/home/a.schnepf/phd/SphericalDreamer/submodules/EGformer/evaluate/INFER_SAMPLE/test_sample/S3D_sample.png"),
        Image.open("/home/a.schnepf/phd/SphericalDreamer/submodules/EGformer/evaluate/INFER_SAMPLE/test_sample/test_sample.png"),
    ]
    outputs = get_egformer_depth(inputs)