import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip

# ----------------------------
# Existing helpers / constants
# ----------------------------

def get_all_image_from_directory(directory, recursive=True):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    exts = {".png"}
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(str(p) for p in it if p.is_file() and p.suffix.lower() in exts)


BASEDIR_LP3D_LUCID = Path("/home/a.schnepf/phd/360-gaussian-splatting/X_ICML_RENDERS/quantitative")
BASEDIR_WONDER_SS_SD = Path("/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/X_ICML_RENDERS/quantitative")

METHODS = [
    "sphericaldreamer",
    "luciddreamer",
    "layerpano3d",
    "wonderjourney",
    "scenescape",
]
TRAJ_TYPES = [
    "trans",
    "rot",
    "rot+trans",
]
EXPNAMES = [
    "dense_rainforest_understory",
    "martian_badlands_2",
    "coral_reef_canyon",
    "phantom_opera_cave_river",
    "sound_of_music_grass_field",
    "upside_down_stranger_things",
]

# ---------------------------------------------------
# IMPORTANT: map each expname to the text prompt used
# ---------------------------------------------------
PROMPTS = {
    "dense_rainforest_understory": (
        "A dense rainforest understory extending forward and behind the observer, "
        "thick overlapping vegetation with no sharp edges, large leaves in deep green tones, "
        "moist ground covered in roots and organic debris, overcast sky filtering light through "
        "the canopy, soft diffused illumination, humid and lush atmosphere, immersive tropical environment."
    ),
    "coral_reef_canyon": (
        "A wide coral reef canyon extending forward and behind the observer, smooth rock walls "
        "covered in coral and marine growth, vibrant yet softened colors including turquoise, "
        "coral pink, and sandy beige, filtered underwater lighting with soft light rays, "
        "floating particles in the water, calm immersive oceanic environment."
    ),
    "martian_badlands_2": (
        "A Martian landscape of rust-red and ochre soil, scattered with dark basalt rocks "
        "and patches of muted green and purple alien vegetation, under a dusty salmon-colored sky."
    ),
    "phantom_opera_cave_river": (
        "A large-scale 3D subterranean cave environment inspired by the Phantom of the Opera underground river setting, "
        "with no visible people or animals, an irregular cavern corridor extending forward and continuing behind the observer, "
        "no sharp edges or well-defined geometry, rounded organic rock formations softened by moisture and time, "
        "a dark slow-moving river running across the ground with reflective surface and subtle ripples, "
        "wet rocky banks with scattered stones and damp sediment, stalactites and stalagmites smoothed into natural shapes, "
        "faint mist hovering above water, dim atmospheric lighting as if from distant unseen lanterns creating soft warm reflections, "
        "deep shadows fading into darkness, cinematic gothic mood, immersive enclosed underground atmosphere with realistic rock and water textures and strong spatial depth."
    ),
    "sound_of_music_grass_field": (
        "A wide open rolling field of lush green grass inspired by The Sound of Music, "
        "gentle natural hills extending forward and continuing behind the observer with no harsh edges or defined geometry, "
        "thick healthy grass blades forming smooth wind-like patterns, scattered wildflower hints subtle and sparse, "
        "cloudy sky overhead with no visible sun, soft diffused daylight creating even illumination and minimal harsh shadows, "
        "distant tree line barely visible on the horizon, peaceful cinematic pastoral mood, "
        "realistic vegetation textures, immersive open countryside atmosphere."
    ),
    "upside_down_stranger_things": (
        "A desolate Upside Down-inspired landscape with cloudy oppressive skies and no visible sun, "
        "the environment stretching forward and behind the observer with organic uneven terrain, "
        "dark damp ground covered in tangled root-like growth and soft alien debris, floating ash-like particles suspended in the air, "
        "twisted vegetation silhouettes without sharp geometry, murky fog reducing visibility in the distance, "
        "muted blue-gray lighting with eerie contrast, wet reflective patches and slimy textures, "
        "cinematic horror mood, immersive otherworldly atmosphere."
    ),
}


# ----------------------------
# CLIP helpers
# ----------------------------

def load_clip_model(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_images_batched(img_paths, model, preprocess, device, batch_size=32):
    """
    Returns normalized CLIP image embeddings of shape [N, D].
    """
    feats = []
    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, dim=0).to(device)
        z = model.encode_image(x)
        z = F.normalize(z, dim=-1)
        feats.append(z.cpu())
    if not feats:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(texts, model, tokenizer, device):
    """
    Returns normalized CLIP text embeddings of shape [N, D].
    """
    tokens = tokenizer(texts).to(device)
    z = model.encode_text(tokens)
    z = F.normalize(z, dim=-1)
    return z.cpu()


def mean_offdiag_cosine(embs: torch.Tensor) -> float:
    """
    Mean pairwise cosine over all ordered pairs excluding diagonal.

    embs: [N, D], assumed normalized.
    """
    n = embs.shape[0]
    if n < 2:
        return math.nan
    sim = embs @ embs.T  # [N, N]
    offdiag_sum = sim.sum() - sim.diag().sum()
    denom = n * (n - 1)
    return float(offdiag_sum / denom)


def mean_clip_score(embs_img: torch.Tensor, emb_txt: torch.Tensor) -> float:
    """
    Average cosine(image, text) over all images in a scene.

    embs_img: [N, D], normalized
    emb_txt:  [D] or [1, D], normalized
    """
    if embs_img.shape[0] == 0:
        return math.nan
    if emb_txt.ndim == 2:
        emb_txt = emb_txt[0]
    scores = embs_img @ emb_txt
    return float(scores.mean())


# ----------------------------
# Main metric computation
# ----------------------------

def compute_clip_tables(
    recursive=True,
    batch_size=32,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
):
    """
    Returns:
      df_exp:
        one row per (method, traj_type, expname)
        columns include:
          - clip_score_exp
          - c_clip_exp
          - n_imgs
      df_method_traj:
        one row per (method, traj_type)
        columns include:
          - avg_clip_score_over_exps
          - avg_c_clip_over_exps
          - c_style_global
      pivot:
        index=method, columns=(metric, traj_type)
    """

    model, preprocess, tokenizer, device = load_clip_model(
        model_name=model_name,
        pretrained=pretrained,
    )

    rows = []
    scene_embeds = {}  # (method, traj_type, expname) -> [N, D]

    jobs = [(m, t, e) for m in METHODS for t in TRAJ_TYPES for e in EXPNAMES]

    for method, traj_type, expname in tqdm(jobs, desc="CLIP (method/traj/exp)", unit="job"):
        base = BASEDIR_LP3D_LUCID if method in ["luciddreamer", "layerpano3d"] else BASEDIR_WONDER_SS_SD
        img_dir = base / traj_type / expname / method / "rgb"
        img_paths = get_all_image_from_directory(img_dir, recursive=recursive)

        if expname not in PROMPTS:
            raise KeyError(f"Missing prompt for expname={expname}")

        if len(img_paths) == 0:
            rows.append({
                "method": method,
                "traj_type": traj_type,
                "expname": expname,
                "n_imgs": 0,
                "img_dir": str(img_dir),
                "clip_score_exp": math.nan,
                "c_clip_exp": math.nan,
            })
            scene_embeds[(method, traj_type, expname)] = torch.empty((0, 0), dtype=torch.float32)
            continue

        # image embeddings
        embs_img = encode_images_batched(
            img_paths=img_paths,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
        )

        # text embedding for this scene prompt
        emb_txt = encode_texts(
            [PROMPTS[expname]],
            model=model,
            tokenizer=tokenizer,
            device=device,
        )[0]

        clip_score_exp = mean_clip_score(embs_img, emb_txt)
        c_clip_exp = mean_offdiag_cosine(embs_img)

        rows.append({
            "method": method,
            "traj_type": traj_type,
            "expname": expname,
            "n_imgs": len(img_paths),
            "img_dir": str(img_dir),
            "clip_score_exp": clip_score_exp,
            "c_clip_exp": c_clip_exp,
        })

        scene_embeds[(method, traj_type, expname)] = embs_img

    df_exp = pd.DataFrame(rows)

    # Aggregate over experiments for CLIP-Score and C-CLIP
    df_method_traj = (
        df_exp.groupby(["method", "traj_type"], as_index=False)
        .agg(
            n_exps_used=("expname", "count"),
            total_imgs=("n_imgs", "sum"),
            avg_clip_score_over_exps=("clip_score_exp", "mean"),
            avg_c_clip_over_exps=("c_clip_exp", "mean"),
        )
    )

    # Compute C-Style globally per (method, traj_type), across all scenes/images
    c_style_rows = []
    for method in METHODS:
        for traj_type in TRAJ_TYPES:
            all_embs = []
            for expname in EXPNAMES:
                z = scene_embeds[(method, traj_type, expname)]
                if z.numel() > 0:
                    all_embs.append(z)

            if len(all_embs) == 0:
                c_style = math.nan
                n_imgs_total = 0
            else:
                Z = torch.cat(all_embs, dim=0)  # [total_imgs, D]
                c_style = mean_offdiag_cosine(Z)
                n_imgs_total = Z.shape[0]

            c_style_rows.append({
                "method": method,
                "traj_type": traj_type,
                "c_style_global": c_style,
                "n_imgs_for_c_style": n_imgs_total,
            })

    df_c_style = pd.DataFrame(c_style_rows)

    df_method_traj = df_method_traj.merge(
        df_c_style,
        on=["method", "traj_type"],
        how="left",
    )

    # Pivot
    pieces = []
    for metric in ["avg_clip_score_over_exps", "avg_c_clip_over_exps", "c_style_global"]:
        p = (
            df_method_traj.pivot(index="method", columns="traj_type", values=metric)
            .reindex(index=METHODS, columns=TRAJ_TYPES)
        )
        p.columns = pd.MultiIndex.from_product([[metric], p.columns])
        pieces.append(p)

    pivot = pd.concat(pieces, axis=1).round(4)

    return df_exp, df_method_traj, pivot


if __name__ == "__main__":
    df_exp, df_method_traj, pivot = compute_clip_tables(
        batch_size=32,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    print("Per-expname table (first rows):")
    print(df_exp.head())

    print("\nOne score per (method, traj_type):")
    print(df_method_traj)

    print("\nPivot (method x (metric, traj_type)):")
    print(pivot)

    outdir = Path("/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/CLIP_METRICS/aggregates")
    outdir.mkdir(parents=True, exist_ok=True)

    df_exp.to_csv(outdir / "clip_df_exp.csv", index=False)
    df_method_traj.to_csv(outdir / "clip_df_method_traj.csv", index=False)
    pivot.to_csv(outdir / "clip_pivot.csv")