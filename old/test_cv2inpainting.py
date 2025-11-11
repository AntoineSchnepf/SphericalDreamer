#!/usr/bin/env python3
# remove_black_grid_inpaint.py
import argparse
import os
import cv2
import numpy as np

def build_mask_from_black(img_bgr, value_thresh=30, chroma_thresh=15):
    """
    Make a binary mask for *near-black* pixels.
    value_thresh: how dark a pixel must be (0..255, lower = darker)
    chroma_thresh: how neutral a pixel must be (max(R,G,B) - min(R,G,B) <= chroma_thresh)
    """
    img = img_bgr.astype(np.int16)
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    val_mask = (mx <= value_thresh)  # darkness
    chroma_mask = ((mx - mn) <= chroma_thresh)  # close to neutral (avoid dark rocks/shadows)
    mask = (val_mask & chroma_mask).astype(np.uint8) * 255
    return mask

def main():
    p = argparse.ArgumentParser(description="Remove thin black grid/lines using OpenCV inpainting.")
    p.add_argument("input", help="Input image (equirectangular panorama or any image)")
    p.add_argument("--mask", help="Optional mask image (white=to inpaint). If omitted, a mask is auto-generated.")
    p.add_argument("--out", default=None, help="Output image path (default: <input>_inpaint.png)")
    p.add_argument("--value-thresh", type=int, default=30, help="Darkness threshold for auto mask (0..255)")
    p.add_argument("--chroma-thresh", type=int, default=15, help="Neutrality threshold for auto mask (0..255)")
    p.add_argument("--dilate", type=int, default=3, help="Dilation kernel size (odd). 0 to disable.")
    p.add_argument("--dilate-iters", type=int, default=1, help="Number of dilation iterations.")
    p.add_argument("--method", choices=["telea", "ns"], default="telea", help="Inpainting algorithm")
    p.add_argument("--radius", type=float, default=3.0, help="Inpainting radius (pixels)")
    args = p.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    # 1) Mask
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f"Could not read mask: {args.mask}")
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        mask = build_mask_from_black(img, args.value_thresh, args.chroma_thresh)

    # 2) (Important) Thicken thin lines so the inpainter "sees" them
    if args.dilate > 0:
        k = max(1, args.dilate)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=args.dilate_iters)

    # 3) Inpaint
    method = cv2.INPAINT_TELEA if args.method == "telea" else cv2.INPAINT_NS
    result = cv2.inpaint(img, mask, inpaintRadius=args.radius, flags=method)

    # 4) Save
    out_path = args.out or os.path.splitext(args.input)[0] + "_inpaint.png"
    cv2.imwrite(out_path, result)
    print(f"Saved: {out_path}")

    # Optional: save the auto mask for inspection
    if not args.mask:
        mask_path = os.path.splitext(args.input)[0] + "_mask.png"
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask preview: {mask_path}")

if __name__ == "__main__":
    main()