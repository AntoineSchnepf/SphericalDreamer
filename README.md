# SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion

### ICML 2026

**Official paper implementation**

> Antoine Schnepf, Karim Kassab, Flavian Vasile, Andrew Comport

| [Project Page](https://sphericaldreamer.github.io/) | [Paper](https://arxiv.org/abs/2605.19974) |

**TL;DR:** SphericalDreamer creates large-scale, fully immersive 3D environments from text by generating and fusing multiple panoramic images into a coherent 3D world.

**Abstract:** *The generation of immersive and navigable 3D environments is increasingly prevalent with the growing adoption of virtual reality and 3D content. However, recent methods face a fundamental limitation: they cannot produce 3D worlds that simultaneously (i) are navigable over long-range spatial extents and (ii) cover the complete omnidirectional field of view (360° horizontally and 180° vertically). To address this challenge, we introduce SphericalDreamer, a method for generating fully immersive and long-range 3D outdoor environments from textual prompts. Our approach is built on the generation of multiple panoramic images, which are subsequently lifted into 3D and fused together while maintaining visual and geometric consistency. SphericalDreamer produces highly detailed, fully immersive 3D environments, while substantially improving scale and navigability compared to prior approaches.*

## Installation

> The installation steps below follow those of [LayerPano3D](https://github.com/YS-IMTech/LayerPano3D), from which we borrow the same Python environment and checkpoints. Thanks to the LayerPano3D authors for the detailed guide.

### 1. Clone the repository

```bash
git clone https://github.com/AntoineSchnepf/SphericalDreamer.git
cd SphericalDreamer
```

### 2. Prepare the environment

```bash
conda create -n sphericaldreamer python==3.9
conda activate sphericaldreamer
```

Install PyTorch (tested on `torch2.4.0+cu118`):

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

Install other requirements:

```bash
pip install -r requirements.txt
pip install timm==0.4.12 --no-deps
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install natten==0.14.4 --trusted-host shi-labs.com -f https://shi-labs.com/natten/wheels/cu118/torch2.4.0/index.html
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```



### 3. Install 360monodepth

This step can be tricky — follow it carefully and open an issue if you run into problems.

**3.1** Edit `submodules/360monodepth/code/cpp/CMakeConfig.txt` line 65 and replace the numpy path with your own.

**3.2** Install system dependencies:

```bash
sudo apt-get install libgtest-dev libeigen3-dev libboost-all-dev libopencv-dev libatlas-base-dev
sudo apt-get install liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev
conda install -c conda-forge libstdcxx-ng=12
```

**3.3** Install pybind11:

```bash
cd submodules/360monodepth/code/cpp/3rd_party
git clone https://github.com/pybind/pybind11.git
cd pybind11 && mkdir build && cd build
cmake .. && make -j8
sudo make install
cd ../../
```

**3.4** Install ceres-solver 1.14.0:

> Note: if the build fails due to a missing `tbb_stddef.h`, replace it with `version.h` on line 224 of `FindTBB.cmake`.

```bash
git clone -b 1.14.0 https://github.com/ceres-solver/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TESTING=OFF \
         -DBUILD_EXAMPLES=OFF \
         -DEIGENSPARSE=ON \
         -DSUITESPARSE=OFF \
         -DCXSPARSE=ON \
         -DGFLAGS=ON
make -j8
sudo make install
cd ../../../
```

**3.5** Build instaOmniDepth:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../python
python setup.py build
python setup.py bdist_wheel
pip install dist/instaOmniDepth-0.1.0-cp39-cp39-linux_x86_64.whl  # check dist/ for the exact filename
```

### 4. Install a renderer

Two renderers are supported for visualizing the generated 3D world:

- **Open3D** (default) — lightweight, lower cost, sufficient for most use cases:
  ```bash
  pip install open3d
  ```
- **Blender** (optional) — required only for highest quality point cloud rendering (`5_render_blender.py`):
  ```bash
  mkdir -p ~/blender && cd ~/blender
  wget https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz
  tar -xf blender-5.0.1-linux-x64.tar.xz
  echo 'export PATH=~/blender/blender-5.0.1-linux-x64:$PATH' >> ~/.bashrc
  source ~/.bashrc
  cd -
  ```
  Then install the required packages into Blender's own Python interpreter:
  ```bash
  BLENDER_PY=$(blender --background --python-expr "import sys; print(sys.executable)" 2>/dev/null | tail -1)
  $BLENDER_PY -m ensurepip --upgrade
  $BLENDER_PY -m pip install --upgrade pip
  $BLENDER_PY -m pip install prodict pyfiglet PyYAML pathlib numpy opencv-python
  ```

### 5. Download checkpoints

```
checkpoints/
├── pano_lora_720*1440_v1.safetensors
├── ControlNetLama.pth
├── sam_vit_h_4b8939.pth
└── depth_anything_v2_vitl.pth
```


| Checkpoint                          | Download                                                                                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pano_lora_720*1440_v1.safetensors` | [HuggingFace](https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA/resolve/main/lora_hubs/pano_lora_720*1440_v1.safetensors?download=true) |
| `ControlNetLama.pth`                | [HuggingFace](https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth?download=true)                                           |
| `sam_vit_h_4b8939.pth`              | [Meta](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)                                                                        |
| `depth_anything_v2_vitl.pth`        | [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)                  |


## Usage

> **Hardware:** The code has been tested on a 40GB A100 GPU. This is required due to the VRAM demands of the Flux image generation models.

Run the full pipeline on one of the provided example configs:

```bash
bash dream.sh examples/lavender_fields.yaml
```

Example configs for various scenes are available in `configs/examples/`. You can also create your own config by editing the prompt and parameters — see `configs/examples/lavender_fields.yaml` as a starting point.

The size of the generated world is controlled by the `num_dreams` entry in the config: a higher value produces a longer, more expansive environment.

## Citation

If you find this research project useful, please consider citing our work:

```bibtex
@article{schnepf2026sphericaldreamer,
  title={{SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion}},
  author={Antoine Schnepf and Karim Kassab and Flavian Vasile and Andrew Comport},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

