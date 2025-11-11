import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def render_eqr_view(
    eqr,                     # HxWxC (uint8 0..255 or float 0..1)
    elevation_n,             # [-1, 1] → pitch in [-π/2, +π/2]
    azimuth_n,               # [-1, 1] → yaw   in [-π,   +π]
    out_h=512, out_w=512,
    hfov_deg=90.0, vfov_deg=None,
):
    H, W = eqr.shape[:2]
    C = 1 if eqr.ndim == 2 else eqr.shape[2]

    yaw   = np.pi * float(azimuth_n)          # [-π, π]
    pitch = (np.pi/2) * float(elevation_n)    # [-π/2, π/2]

    if vfov_deg is None:
        vfov_deg = hfov_deg * (out_h / out_w)

    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    # Intrinsics (principal point at center)
    fx = (out_w / 2) / np.tan(hfov / 2)
    fy = (out_h / 2) / np.tan(vfov / 2)
    cx = (out_w - 1) / 2
    cy = (out_h - 1) / 2

    # Pixel grid
    x = np.arange(out_w)
    y = np.arange(out_h)
    xx, yy = np.meshgrid(x, y)

    # Rays in camera-local coordinates (z forward, x right, y down)
    Xc = (xx - cx) / fx
    Yc = (yy - cy) / fy   # positive downward
    Zc = np.ones_like(Xc)

    # Build camera basis from yaw (azimuth) and pitch (elevation)
    # Forward (look) direction with world-up = +Y
    forward = np.array([
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw) * np.cos(pitch),
    ], dtype=np.float64)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # If near the pole, tweak up to avoid degeneracy
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)  # already normalized

    # Compose world rays: dir = X*right + (-Y)*up + Z*forward (note -Y because image y is down)
    dir_x = Xc[..., None] * right[None, None, :]
    dir_y = (-Yc[..., None]) * up[None, None, :]
    dir_z = Zc[..., None] * forward[None, None, :]
    dirs = dir_x + dir_y + dir_z

    # Normalize
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Convert to (theta, phi) with Y as up
    Xw, Yw, Zw = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = np.arctan2(Xw, Zw)                      # [-π, π]
    phi   = np.arcsin(np.clip(Yw, -1.0, 1.0))       # [-π/2, π/2]

    # Equirectangular mapping (u right, v down)
    u = (theta + np.pi) / (2 * np.pi) * W
    v = (-phi + np.pi/2) / np.pi * H   # upright: north/top at v≈0

    # Wrap/clamp
    u = np.mod(u, W)
    v = np.clip(v, 0, H - 1.000001)

    # Sample (OpenCV if available; NumPy fallback)
    try:
        import cv2
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        eqr_in = eqr
        to_uint8 = False
        if eqr.dtype != np.uint8:
            eqr_in = (np.clip(eqr, 0, 1) * 255).astype(np.uint8)
            to_uint8 = True
        view = cv2.remap(eqr_in, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if not to_uint8:
            return view
        return (view.astype(np.float32) / 255.0).astype(eqr.dtype, copy=False)
    except ImportError:
        # Bilinear sampling in NumPy
        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        u1 = (u0 + 1) % W
        v1 = np.clip(v0 + 1, 0, H - 1)

        du = (u - u0)[..., None]
        dv = (v - v0)[..., None]

        def gather(ii, jj):
            if C == 1:
                return eqr[jj, ii][..., None]
            return eqr[jj, ii, :]

        I00 = gather(u0, v0)
        I10 = gather(u1, v0)
        I01 = gather(u0, v1)
        I11 = gather(u1, v1)

        view = (1-du)*(1-dv)*I00 + du*(1-dv)*I10 + (1-du)*dv*I01 + du*dv*I11
        return view.astype(eqr.dtype, copy=False)


if __name__ == '__main__':
    # eqr: your HxWx3 panorama (uint8 0..255 or float 0..1)
    # Looking straight ahead: elevation_n=0, azimuth_n=0
    eqr_pil = Image.open("/home/a.schnepf/phd/SphericalDreamer/SphericalDreamerRecurse_outputs/beach-bkp/dream_00/01_pano_rgb.png")
    eqr = np.array(eqr_pil)
    view = render_eqr_view(eqr, elevation_n=0.0, azimuth_n=0.0, out_h=600, out_w=800, hfov_deg=90)

    # Look 45° up and 90° right:
    view2 = render_eqr_view(eqr, elevation_n=+0.5, azimuth_n=+0.5)

    # If you want a wider or narrower view:
    view_wide = render_eqr_view(eqr, 0.0, 0.0, hfov_deg=120)
    view_narrow = render_eqr_view(eqr, 0.0, 0.0, hfov_deg=60)

    plt.imshow(view)