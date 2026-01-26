from PIL import Image

imgpath_in = "/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/X_ICML_RENDERS/outside/phantom_opera_cave_river_outside/wonderjourney/rgb_eqr/x=-0.00/azi=0.00_rgba.png"
imgpath_out = imgpath_in.replace(".png", "_crop400.png")

img = Image.open(imgpath_in)
w, h = img.size

# crop: (left, upper, right, lower)
crop_px = 400
img_cropped = img.crop((
    crop_px*2,
    crop_px*1,
    w - crop_px*4,
    h - crop_px*2
))

img_cropped.save(imgpath_out)