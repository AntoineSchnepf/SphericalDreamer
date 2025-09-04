# local imports
import sys
from src.lama import LamaInpainting
import my_utils
from PIL import Image
import numpy as np
lama_model = LamaInpainting()


image = Image.open("SphericalDreamerRecurse_outputs/city/dream_00/01_pano_rgb.png").convert('RGB')
image_arr = np.array(image)

mask = np.zeros(shape=(image.size[1], image.size[0])).astype("bool")
mask[50:200, 50:200] = 1
image_arr[mask, None] = 0
image = Image.fromarray(image_arr)
mask_pil = my_utils.numpy_to_PIL(mask).convert('L')

tmp_size = image.size
output = lama_model(image, mask_pil)
output_image = Image.fromarray(output)
output_image