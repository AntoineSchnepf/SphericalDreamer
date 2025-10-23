import os
import sys
import cv2
from matplotlib import image
from src.pipeline_flux import FluxPipeline
from src.pipeline_flux_fill import FluxFillPipeline
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import torch
import numpy as np
from PIL import Image, ImageOps
import copy
from functools import partial
from skimage.segmentation import find_boundaries
from scipy.ndimage import maximum_filter, minimum_filter
import logging
import matplotlib.pyplot as plt
import time
import pickle as pkl
import argparse
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from utils.depth_alignment import Pano_depth_estimation
import my_utils

logging.disable(logging.CRITICAL + 1)


FAR=10
NEAR=0.1

class PointCloud:
    def __init__(self, pts, colors):
        """
        pts: np.array of shape [..., 3]
        colors: np.array of shape [..., 3] with values in [0-1]
        """
        self.pts = pts.reshape(-1, 3)
        self.colors = colors.reshape(-1, 3)
        assert self.pts.shape[0] == self.colors.shape[0], "Error: pts and colors must have the same number of points"

    def get_o3d_pointcloud(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd

class SphereState:
    def __init__(self, pts_carte, colors, pose):
        """everything in spherical coordinates"""
        self.pts = pts_carte
        self.colors = colors
        self.pose = pose
        self.is_world_pcd_init=False
    
    def init_world_pcd(self):
        assert not self.is_world_pcd_init, "World pointcloud is already initialized"
        assert self.pose is not None, "Pose must be defined to compute world pointcloud"

        self.world_pcd = PointCloud(
            pts=my_utils.cam_carte2world_3D(self.pts, self.pose),
            colors=self.colors
        )
        self.is_world_pcd_init=True

    def get_world_pcd(self):
        "returns pointcloud in world coordinates"

        if not self.is_world_pcd_init:
            self.init_world_pcd()

        return self.world_pcd

    def update_pose(self, new_pose):
        self.pose = new_pose
        self.is_world_pcd_init=False

_default_opening_kwargs = {
    'opening_mode': 'cut+cylinder',
    'delta_cut': 2*np.pi/3,
    'cut_distance_percentile': 90,
}
class Sphere:

    def __init__(self, pose, pts_carte, colors, forward_sph=None, forward_carte=None, opening_kwargs=_default_opening_kwargs):
        """
        Can be derived in four different ways: open left, open right, open both, open none
        Input points are expected not to be opened ye.
        A sphere has a forward direction, which can be expressed in spherical coordinates or cartesian coordinates
        """
        assert (forward_sph is not None) or (forward_carte is not None), "Error: forward direction must be provided in either spherical or cartesian coordinates"
        self.forward_sph = forward_sph if forward_sph is not None else my_utils.carte2sph_3D(forward_carte)
        self.forward_carte = forward_carte if forward_carte is not None else my_utils.sph2carte_3D(forward_sph)
        self.opening_kwargs = opening_kwargs
        self.pose = pose

        # open the Sphere in all four ways to get all states
        t = time.time()
        self.init_states(pts_carte, colors)
        print(f"Sphere init took {time.time() - t:.1f}s")

    @staticmethod
    def filter_nan(pts_carte, colors):
        mask_finite = np.isfinite(pts_carte).all(axis=-1) & np.isfinite(colors).all(axis=-1)
        pts_carte = pts_carte[mask_finite]
        colors = colors[mask_finite]
        return pts_carte, colors
    
    def init_states(self, pts_carte, colors):
        # filter nan
        self.pts_carte, self.colors = self.filter_nan(pts_carte, colors)
        # compute all openings
        self.closed = self._close(self.pts_carte, self.colors)
        self.both_opened = self._open_both(self.pts_carte, self.colors)
        self.left_opened = self._open_left(self.pts_carte, self.colors)
        self.right_opened = self._open_right(self.pts_carte, self.colors)

    def get_state(self, open_left, open_right):
        if open_left and open_right:
            return self.both_opened_sphere
        elif open_left and not open_right:
            return self.left_opened_sphere
        elif not open_left and open_right:
            return self.right_opened_sphere
        else:
            return self.closed_sphere
    
    def _close(self, pts_carte, colors):
        sphere_closed = SphereState(
            pts_carte=pts_carte, 
            colors=colors,
            pose=self.pose
        )
        return sphere_closed
    
    def _open_left(self, pts_carte, colors):
        _, open_pts_carte, mask_opening, cut_distance = my_utils.open_world(
            forward_carte=-self.forward_carte,
            pts_carte=pts_carte,
            **self.opening_kwargs
        )
        sphere_opened_left = SphereState(
            pts_carte=open_pts_carte[mask_opening], 
            colors=colors[mask_opening],
            pose=self.pose
        )
        self.half_size_left = cut_distance
        return sphere_opened_left

    def _open_right(self, pts_carte, colors):
        _, open_pts_carte, mask_opening, cut_distance = my_utils.open_world(
            forward_carte=self.forward_carte,
            pts_carte=pts_carte,
            **self.opening_kwargs
        )
        sphere_opened_right = SphereState(
            pts_carte=open_pts_carte[mask_opening], 
            colors=colors[mask_opening],
            pose=self.pose
        )
        self.half_size_right = cut_distance
        return sphere_opened_right

    def _open_both(self, pts_carte, colors):
        _, open_pts_carte, mask_opening1, cut_distance1 = my_utils.open_world(
            forward_carte=self.forward_carte,
            pts_carte=pts_carte,
            **self.opening_kwargs
        )
        open_pts_carte = open_pts_carte[mask_opening1]
        colors = colors[mask_opening1]

        _, open_pts_carte, mask_opening2, cut_distance2 = my_utils.open_world(
            forward_carte=-self.forward_carte,
            pts_carte=open_pts_carte,
            **self.opening_kwargs
        )
        open_pts_carte = open_pts_carte[mask_opening2]
        colors = colors[mask_opening2]
        sphere_opened_both = SphereState(
            pts_carte=open_pts_carte, 
            colors=colors,
            pose=self.pose
        )

        return sphere_opened_both

    def add_new_points(self, new_pts_carte, new_colors):
        pts_carte = np.concatenate((self.pts_carte, new_pts_carte.reshape(-1, 3)), axis=0)
        colors = np.concatenate((self.colors, new_colors.reshape(-1, 3)), axis=0)
        self.init_states(pts_carte, colors)

    def update_pose(self, new_pose):
        self.pose = new_pose
        for state in [self.closed, self.both_opened, self.left_opened, self.right_opened]:
            state.update_pose(new_pose)

class Regression1D:

    @staticmethod
    def fit_nw_grid_interpolator_1d(X, Y, bandwidth, grid_size=1024, margin=3.0):
        """
        Fit-time:
        - builds a 1D grid that extends beyond data by `margin * bandwidth`,
        - evaluates Nadaraya–Watson (Gaussian) on that grid,
        - returns an inference-only f(x) that linearly interpolates on the grid.

        Parameters
        ----------
        X : array-like, shape (n,)
            Training inputs.
        Y : array-like, shape (n,)
            Training targets.
        bandwidth : float
            Gaussian kernel width (σ). Larger => smoother.
        grid_size : int, default 1024
            Number of grid points to precompute.
        margin : float, default 3.0
            Extra range (in units of σ) added on both sides of [min(X), max(X)]
            to stabilize edge behavior and improve clamped extrapolation.

        Returns
        -------
        f : callable
            Inference-only function. Accepts batched x with shape [...] and returns shape [...].
            Interpolates within the grid; clamps to edge values outside the grid.
        """
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        assert X.ndim == 1 and Y.ndim == 1 and X.size == Y.size, "X and Y must be 1D and same length."
        assert bandwidth > 0.0, "bandwidth must be positive."

        # Build grid with padding to mitigate boundary artifacts
        x_min = X.min() - margin * bandwidth
        x_max = X.max() + margin * bandwidth
        x_grid = np.linspace(x_min, x_max, int(grid_size))

        # Evaluate NW smoother on the grid (vectorized, O(grid_size * n))
        D = (x_grid[:, None] - X[None, :]) / bandwidth                  # shape (G, n)
        W = np.exp(-0.5 * D**2)                                         # Gaussian kernels
        W_sum = W.sum(axis=1) + 1e-12                                   # avoid divide-by-zero
        y_grid = (W @ Y) / W_sum                                        # shape (G,)

        # Inference-only interpolator: piecewise-linear + clamped extrapolation
        def f(x):
            x = np.asarray(x, dtype=float)
            x_flat = x.ravel()
            # np.interp clamps to left/right values if outside the grid
            y_flat = np.interp(x_flat, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])
            return y_flat.reshape(x.shape)

        return f

    @staticmethod
    def fit_local_min_knots_interpolator_1d(
        X, Y, bandwidth, *, handle_empty="skip", tie_break="center"
    ):
        
        """
        One-shot: build (X_min, Y_min) knots per bin, and return an
        inference-only monotone interpolator.
        """
                
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        if X.size != Y.size:
            raise ValueError("X and Y must have the same length.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("X and Y must be finite.")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0.")

        x_min, x_max = X.min(), X.max()
        n_bins = max(1, int(np.ceil((x_max - x_min) / bandwidth)))
        bin_edges = x_min + np.arange(n_bins + 1) * bandwidth
        if bin_edges[-1] < x_max:
            bin_edges[-1] = x_max

        bin_idx = np.digitize(X, bin_edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        X_k = np.full(n_bins, np.nan)
        Y_k = np.full(n_bins, np.nan)

        for b in range(n_bins):
            mask = (bin_idx == b)
            if not np.any(mask):
                continue
            Xb = X[mask]
            Yb = Y[mask]
            y_min = np.min(Yb)
            tie_mask = (Yb == y_min)
            if tie_break == "first":
                i = np.argmax(tie_mask)
            elif tie_break == "last":
                i = len(Yb) - 1 - np.argmax(tie_mask[::-1])
            else:  # "center"
                center = 0.5 * (bin_edges[b] + bin_edges[b + 1])
                idxs = np.flatnonzero(tie_mask)
                i = idxs[np.argmin(np.abs(Xb[idxs] - center))]
            X_k[b] = Xb[i]
            Y_k[b] = y_min

        if handle_empty == "nearest":
            # forward fill
            for i in range(1, n_bins):
                if not np.isfinite(Y_k[i]):
                    X_k[i] = X_k[i - 1]
                    Y_k[i] = Y_k[i - 1]
            # backward fill
            for i in range(n_bins - 2, -1, -1):
                if not np.isfinite(Y_k[i]):
                    X_k[i] = X_k[i + 1]
                    Y_k[i] = Y_k[i + 1]

        finite = np.isfinite(X_k) & np.isfinite(Y_k)
        X_k = X_k[finite]
        Y_k = Y_k[finite]
        if X_k.size == 0:
            def f_nan(x):
                x = np.asarray(x, dtype=float)
                return np.full_like(x, np.nan)
            return f_nan, X_k, Y_k

        order = np.argsort(X_k)
        X_k = X_k[order]
        Y_k = Y_k[order]

        uniq_x, inv = np.unique(X_k, return_inverse=True)
        if uniq_x.size != X_k.size:
            y_min_by_x = np.full_like(uniq_x, np.inf, dtype=float)
            np.minimum.at(y_min_by_x, inv, Y_k)
            X_k, Y_k = uniq_x, y_min_by_x

        def f(x):
            x = np.asarray(x, dtype=float)
            x_flat = x.ravel()
            y_flat = np.interp(x_flat, X_k, Y_k, left=Y_k[0], right=Y_k[-1])
            return y_flat.reshape(x.shape)

        return f, X_k, Y_k

    @staticmethod
    def _isotonic_l2_pav(y, w=None):
        """
        Pool-Adjacent-Violators for nondecreasing isotonic regression (L2).
        Returns the closest (in weighted L2) nondecreasing vector to y.
        """
        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            return y
        if w is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(w, dtype=float)
        # Initialize blocks
        v = y.copy()
        wv = w.copy()
        # Stack of block end indices
        end = [0]
        for i in range(n):
            v[i] = y[i]
            wv[i] = w[i]
            end.append(i + 1)
            # Merge while violating monotonicity
            while len(end) >= 3:
                i2 = end[-1]      # end of last block
                i1 = end[-2]      # start of last block
                i0 = end[-3]      # start of penultimate block
                if v[i1 - 1] <= v[i2 - 1]:
                    break
                # pool blocks [i0:i1] and [i1:i2]
                tot_w = wv[i1 - 1] + wv[i2 - 1]
                avg = (wv[i1 - 1] * v[i1 - 1] + wv[i2 - 1] * v[i2 - 1]) / tot_w
                v[i1 - 1] = avg
                wv[i1 - 1] = tot_w
                # pop last block
                end.pop()
                end[-1] = i2  # extend previous block to new end
        # Expand block means
        y_iso = np.empty_like(y)
        start = 0
        for e in end[1:]:
            y_iso[start:e] = v[e - 1]
            start = e
        return y_iso

    @staticmethod
    def _make_monotone_increasing_from_knots(
        X_knots, Y_knots, *, weights=None, lower=None, upper=None,
        strict=False, eps=1e-12
    ):
        """
        Enforce nondecreasing Y over X_knots via isotonic regression, then
        return an inference-only linear interpolator over (X_knots, Y_iso).

        - 'sticks' to Y_knots wherever they already satisfy monotonicity.
        - optional bounds 'lower'/'upper' clamp the final curve.
        - if strict=True, nudges flat segments by tiny eps to be strictly increasing.

        Returns f_mono, (X_knots_sorted, Y_iso)
        """
        Xk = np.asarray(X_knots, dtype=float)
        Yk = np.asarray(Y_knots, dtype=float)
        mask = np.isfinite(Xk) & np.isfinite(Yk)
        Xk, Yk = Xk[mask], Yk[mask]
        if Xk.size == 0:
            def f_nan(x):
                x = np.asarray(x, dtype=float)
                return np.full_like(x, np.nan)
            return f_nan, (Xk, Yk)

        order = np.argsort(Xk)
        Xk = Xk[order]
        Yk = Yk[order]
        if weights is not None:
            w = np.asarray(weights, dtype=float)[mask][order]
        else:
            w = None

        Y_iso = Regression1D._isotonic_l2_pav(Yk, w=w)

        if lower is not None:
            Y_iso = np.maximum(Y_iso, lower)
        if upper is not None:
            Y_iso = np.minimum(Y_iso, upper)

        if strict:
            # Make strictly increasing by adding tiny offsets to flat runs
            # while staying within bounds if provided.
            diffs = np.diff(Y_iso)
            flat_idx = np.where(diffs <= 0)[0]
            k = 0
            for i in flat_idx:
                k += 1
                Y_iso[i + 1] = max(Y_iso[i + 1], Y_iso[i] + eps)
            # optional: re-clip
            if lower is not None:
                Y_iso = np.maximum(Y_iso, lower)
            if upper is not None:
                Y_iso = np.minimum(Y_iso, upper)

        def f(x):
            x = np.asarray(x, dtype=float)
            x_flat = x.ravel()
            y_flat = np.interp(x_flat, Xk, Y_iso, left=Y_iso[0], right=Y_iso[-1])
            return y_flat.reshape(x.shape)

        return f, (Xk, Y_iso)

    @staticmethod
    def fit_local_min_knots_monotone_interpolator_1d(
        X, Y, bandwidth, *, handle_empty="skip", tie_break="center",
        lower=None, upper=None, strict=False, weights=None
    ):
        """
        One-shot: build (X_min, Y_min) knots per bin, then enforce
        nondecreasing Y via isotonic regression, and return an
        inference-only monotone interpolator.
        """
        _, Xk, Yk = Regression1D.fit_local_min_knots_interpolator_1d(
            X, Y, bandwidth, handle_empty=handle_empty, tie_break=tie_break
        )
        f_mono, (Xk_sorted, Y_iso) = Regression1D._make_monotone_increasing_from_knots(
            Xk, Yk, weights=weights, lower=lower, upper=upper, strict=strict
        )
        
        return f_mono

class GeometryTransforms:

    @staticmethod
    def depth_transform(
        D_raw: np.ndarray,
        n: float = 0.5,      # near (meters)
        f: float = 200.0,    # far  (meters)
        method: str = "inv",  # "inv", "exp", "gamma", 'threshold'
        gamma: float = 0.6,  # used if method=="gamma" (gamma<1 expands far)
        k: float = 3.0,       # used if method=="exp"   (larger k → more far expansion)
        plot: bool = False
    ) -> np.ndarray:
        """
        Map depth D in [0,1] (near≈0, far≈1) to metric range Z in [n,f], monotonically increasing.

        Methods:
        - "linear":   Z = n + D*(f-n)  (identity, no correction)
        - "inv":  assumes D is ~linear in 1/Z but with near→0, far→1.
                        Z = 1 / ( 1/n - D*(1/n - 1/f) )
                        (Excellent default to remove far stacking.)
        - "exp":       convex exponential easing toward f:
                        s = (exp(k*D)-1)/(exp(k)-1); Z = n + s*(f-n)
                        (k>0; increases separation at large D.)
        - "gamma":     gamma pre-warp then linear:
                        Dg = D**gamma (gamma<1 expands far); Z = n + Dg*(f-n)
        """
        D = np.asarray(D_raw, dtype=np.float32)
        D = np.clip(D, 0.0, 1.0)
        
        if plot: 
            plot_d = np.linspace(0, 1, 500)

        if method == "linear":
            Z = D * (f - n) + n
            if plot:
                Z_plot = plot_d * (f - n) + n

        elif method =='threshold':
            def corr(D):
                D[D>0.9] = f
                return D
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        elif method == "inv":
            n = max(n, 1e-3)  # avoid div-by-zero
            def corr(D):
                denom = (1.0 / n) - D * (1.0 / n - 1.0 / f)
                Z = 1.0 / np.clip(denom, 1e-9, None)
                return Z
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        elif method == "exp":
            def corr(D):
                s = (np.exp(k * D) - 1.0) / (np.exp(k) - 1.0 + 1e-9)
                Z = n + s * (f - n)
                return Z
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        elif method == "gamma":
            def corr(D):
                Dg = D ** gamma   # gamma<1 expands high-D region
                Z = n + Dg * (f - n)
                return Z
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        else:
            raise ValueError(f"Unknown method: {method}")

        if plot:
            plt.figure()
            plt.plot(plot_d, Z_plot, label='corrected')
            plt.plot(plot_d, plot_d * (f - n) + n, '--', label='linear')
            plt.legend()
            plt.xlabel("Input D (0=near, 1=far)")
            plt.ylabel("Output Z (meters)")
            plt.title(f"Depth Linearization: method={method}, n={n}, f={f}")
            plt.grid()
            plt.show()
        # Optional: set invalid/zero inputs to NaN
        Z[~np.isfinite(D)] = np.nan
        return Z

    @staticmethod
    def _l2_errors(x, y):
        "norm over last axis"
        return np.sqrt((x - y)**2)

    @staticmethod
    def _l1_errors(x, y):
        "norm over last axis"
        return np.abs(x - y)

    @staticmethod
    def correct_floor_v1(P, depth_map_eqr, error_type='l1', plot=False):
        """
        Correct points using trigonometry and an heuristic to make the floor flat
        The next version is better. Keeping this one just for reference.
        """
        thetas = my_utils.get_canonical_sph_pixels(height, width)[..., 0]
        avg_depth_vertical = np.nanmean(depth_map_eqr, axis=1)  # [H, ]
        r_horizon_theta_range=(
            np.deg2rad(-10), np.deg2rad(-1) 
        )
        r_horizon_band_mask = (thetas[:,0] >= r_horizon_theta_range[0]) & (thetas[:,0] <= r_horizon_theta_range[1])
        r_horizon = np.nanmean(avg_depth_vertical[r_horizon_band_mask])  # scalar

        if error_type=='l2':
            strength = GeometryTransforms._l2_errors(depth_map_eqr, r_horizon)
        elif error_type=='l1':
            strength = GeometryTransforms._l1_errors(depth_map_eqr, r_horizon)
        else:
            raise ValueError(f"Unknown error type: {error_type}. Choose from 'l1' or 'l2'.")
        
        strength = (strength - np.min(strength)) / (np.max(strength) - np.min(strength) + 1e-8)
        strength[thetas >= 0] = 0.0

        correction_raw =  (avg_depth_vertical[:, None] * np.cos(thetas + np.pi/2))[..., None] * np.array([0, 0, 1])
        correction = strength[..., None] * correction_raw
        corrected_pts = P + correction

        if plot:
            fig, axes = plt.subplots(3,2, figsize=(8,16))

            axes[0,0].set_title("Depth")
            axes[0,0].imshow(depth_map_eqr)
            fig.colorbar(axes[0,0].imshow(depth_map_eqr), ax=axes[0,0])

            axes[0,1].set_title("Depth Correction")
            axes[0,1].imshow(correction[..., 2])
            fig.colorbar(axes[0,1].imshow(correction[..., 2]), ax=axes[0,1])
            
            axes[1,0].set_title("Correction Raw")
            axes[1,0].imshow(correction_raw)
            fig.colorbar(axes[1,0].imshow(correction_raw[..., 2]), ax=axes[1,0])


            axes[1,1].set_title("Correction Strength")
            axes[1,1].imshow(strength)
            fig.colorbar(axes[1,1].imshow(strength), ax=axes[1,1])

            axes[2,0].set_title("Depth and Correction Profile")
            correction_profile = np.nanmean(correction[..., 2], axis=1)
            axes[2,0].plot(thetas[:,0], correction_profile, label="Average depth correction")
            theta_band = np.nanmean(thetas, axis=1)
            avg_depth_vertical = np.nanmean(depth_map_eqr, axis=1)
            axes[2,0].plot(theta_band, avg_depth_vertical, label="average depth (before)")
            axes[2,0].legend()
            axes[2,0].set_xlabel("Elevation (radians)")
            axes[2,0].set_ylabel("Average Depth")

            axes[2,1].set_title("Z value Profiles before/after correction")
            z_before = P[..., 2].mean(axis=1)
            axes[2,1].plot(theta_band, z_before, label="Before correction")
            axes[2,1].set_xlabel("Elevation (radians)")
            axes[2,1].set_ylabel("Average Z value")

            z_after = corrected_pts[..., 2].mean(axis=1)
            axes[2,1].plot(theta_band, z_after, label="After correction")
            axes[2,1].legend()
            plt.tight_layout()
            plt.show()

            # show z axis 
            fig, axes = plt.subplots(2,1)
            fig.suptitle("Z values before/after correction")
            axes[0].set_title("Before Correction")
            im0 = axes[0].imshow(P[..., 2], vmin=-1, vmax=1)
            fig.colorbar(im0, ax=axes[0])
            axes[1].set_title("After Correction")
            im1 = axes[1].imshow(corrected_pts[..., 2], vmin=-1, vmax=1)
            fig.colorbar(im1, ax=axes[1])
            plt.tight_layout()
            plt.show()


        return corrected_pts, correction, correction_raw, strength

    @staticmethod
    def correct_floor(P, plot=True):
        """
        Corrects a 3D point cloud so that the detected floor becomes flat.

        This function identifies and models the lowest (floor) surface of a 
        point cloud, assuming the floor corresponds to the lower envelope 
        of the (Z, sqrt(X² + Y²)) profile below the horizon. It fits a 
        non-decreasing local-minimum interpolator to estimate the floor height 
        as a function of radial distance from the origin, and then vertically 
        adjusts all points so that this estimated floor becomes flat.

        Parameters
        ----------
        P : ndarray of shape (..., 3)
            Input 3D point cloud. Each row or voxel corresponds to (X, Y, Z)
            coordinates in world or sensor space.
        plot : bool, optional, default=True
            If True, plots the original (C, Z) data below the horizon and the 
            fitted floor profile for visual inspection.

        Returns
        -------
        P_corrected : ndarray of shape (..., 3)
            The point cloud after applying the vertical correction that flattens 
            the floor. The Z-axis is adjusted such that the estimated floor 
            becomes level at the mean horizon height.
        correction_raw : ndarray of shape (..., 3)
            The vertical displacement applied to each point (ΔX, ΔY, ΔZ). 
            Only the Z component is non-zero.

        Notes
        -----
        The correction is derived as follows:

        1. Convert the 3D points (X, Y, Z) to polar coordinates 
        `C = sqrt(X² + Y²)` and select only the region below the horizon.
        2. Fit a smooth, non-decreasing local-minimum interpolator 
        `Z_floor(C)` to the lowest observed points in each radial bin.
        This ensures that the estimated floor height is monotonic with 
        respect to distance and not affected by local noise or clutter.
        3. Estimate the mean horizon height `Z_horizon` using points near 
        the horizon band.
        4. Compute the per-point correction along Z as 
        `ΔZ = Z_horizon - Z_floor(C)`, and apply it to flatten the floor.

        The result is a geometrically corrected point cloud where the 
        ground plane appears level, useful for visualization, mapping, 
        or downstream processing.

        See Also
        --------
        Regression1D.fit_local_min_knots_monotone_interpolator_1d : 
            Used internally to fit the monotone local-minimum interpolator.
        my_utils.get_canonical_sph_pixels : 
            Provides the spherical pixel mapping used to select points 
            below the horizon.
        """
        sph_canon = my_utils.get_canonical_sph_pixels(height, width)
        phi_range = (
            np.deg2rad(0), np.deg2rad(360)
        )
        below_horizon_mask_and_phi_in_range = (sph_canon[..., 0] < 0) & (sph_canon[..., 1] >= phi_range[0]) & (sph_canon[..., 1] <= phi_range[1])
        X_bh, Y_bh, Z_bh = P[below_horizon_mask_and_phi_in_range].T
        c_bh = np.sqrt(X_bh**2 + Y_bh**2)

        bandwidth = 0.05 * (np.max(c_bh) - np.min(c_bh)) 
        t0 = time.time()
        f_approx = Regression1D.fit_local_min_knots_monotone_interpolator_1d(c_bh, Z_bh, bandwidth=bandwidth)
        print("Regression for depth correction took:", time.time() - t0)
        
        # compute correction
        thetas = sph_canon[..., 0]
        horizon_theta_range=(
            np.deg2rad(-10), np.deg2rad(-1) 
        )
        horizon_band_mask = (thetas[:,0] >= horizon_theta_range[0]) & (thetas[:,0] <= horizon_theta_range[1])
        les_z_in_band = P[horizon_band_mask, ..., 2]
        z_horizon = np.nanmean(les_z_in_band)

        correction_raw = np.zeros_like(P)
        c_input = np.sqrt(P[...,0]**2 + P[...,1]**2)
        correction_raw = z_horizon - f_approx(c_input)
        correction_raw = correction_raw[..., None] * np.array([0, 0, 1])
        P_corrected = P + correction_raw

        if plot:
            plt.figure()
            plt.title("Z values below horizon")
            plt.scatter(c_bh, Z_bh, s=1, label='Data Points')
            les_c = np.linspace(np.min(c_bh), np.max(c_bh), 100)

            les_z_approx = f_approx(les_c)
            plt.plot(les_c, les_z_approx, color='r', label='Kernel Regression Fit')
            plt.xlabel("C = sqrt(X^2 + Y^2)")
            plt.ylabel("Z values")
            plt.legend()
            plt.show()

            plt.figure()
            plt.imshow(correction_raw[..., 2], cmap='jet')
            plt.title("Raw Depth Correction (new method)")
            plt.colorbar()
            plt.show()
        
        return P_corrected

    @staticmethod
    def get_sky_mask(
            depth_map,         
            thetas_range_for_sky_detection = (np.deg2rad(80), np.deg2rad(90)),
            eps = 0.5
        ):

        thetas = my_utils.get_canonical_sph_pixels(height, width)[..., 0]
        sky_theta_mask = (thetas >= thetas_range_for_sky_detection[0]) & (thetas <= thetas_range_for_sky_detection[1])
        depth_sky_values = depth_map[sky_theta_mask]
        threshold = np.nanmean(depth_sky_values) - eps * np.nanstd(depth_sky_values)
        sky_mask = depth_map >= threshold
        return sky_mask

    @staticmethod
    def _smoothstep(a, b, x):
        """Cubic smoothstep from 0→1 on [a,b]."""
        t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def correct_walls(
            pts_sph,
            theta_range,
            edge=np.deg2rad(8.0),
            sky_mask= None    
        ):
        """
        Smoothly map points from sphere to cylinder only within an elevation band,
        with soft transitions near the edges.

        Conventions:
        - phi: azimuth in [0, 2π)
        - theta  : elevation in [-π/2, π/2] (0 at equator)
        - r    : radius of sphere/cylinder
        - phi_range: (phi_min, phi_max) where mapping is 'active'
        - edge : half-width (radians) of the soft transition at each band edge
        - sky_mask: optional boolean mask where sky points are untouched
        Returns:
        x_out, y_out, z_out : arrays of mapped Cartesian points
        w                   : blend weight in [0,1] (0=sphere, 1=cylinder)
        """
        theta_min, theta_max = theta_range

        theta, phi, r = pts_sph[..., 0], pts_sph[..., 1], pts_sph[..., 2]

        # Sphere coords (elevation convention)
        x_s = r * np.cos(theta) * np.cos(phi)
        y_s = r * np.cos(theta) * np.sin(phi)
        z_s = r * np.sin(theta)

        # Cylinder coords (same z)
        x_c = r * np.cos(phi)
        y_c = r * np.sin(phi)
        z_c = z_s  # unchanged

        # Build a smooth "band" weight:
        #  - ramp up across lower edge:  smoothstep(theta_min - edge, theta_min + edge, theta)
        #  - ramp down across upper edge: 1 - smoothstep(theta_max - edge, theta_max + edge, theta)
        w_up   = GeometryTransforms._smoothstep(theta_min - edge, theta_min + edge, theta)
        w_down = 1.0 - GeometryTransforms._smoothstep(theta_max - edge, theta_max + edge, theta)
        w = np.clip(w_up * w_down, 0.0, 1.0)

        # Blend between sphere (0) and cylinder (1)
        x_out = (1.0 - w) * x_s + w * x_c
        y_out = (1.0 - w) * y_s + w * y_c
        z_out = z_s  # identical in both, so blending unnecessary; kept for clarity

        # Optionally preserve sky points from modification
        if sky_mask is not None:
            x_out[sky_mask] = x_s[sky_mask]
            y_out[sky_mask] = y_s[sky_mask]
            z_out[sky_mask] = z_s[sky_mask]

        pts_corrected_carte = np.stack((x_out, y_out, z_out), axis=-1)
        return pts_corrected_carte
 
    @staticmethod
    def run_corrective_pipeline(depth, sphere_radius, correct_floor, correct_walls, remove_sky, indoor_or_outdoor, verbose=False):
        #TODO:
        # - docstring

        assert not np.any(np.isnan(depth)), "Depth contains NaNs!"
        assert indoor_or_outdoor in ['indoor', 'outdoor'], "indoor_or_outdoor must be either 'indoor' or 'outdoor'"

        # 1. Get Metric Depth
        depth_corrected = GeometryTransforms.depth_transform(
            depth, 
            method="inv", 
            n=NEAR, 
            f=FAR, 
            gamma=5,
            plot=True
        )
        if verbose:
            print("a. Metric Depth Obtained.")
            
        # 2. Project to Camera Space in Cartesian Coordinates
        pts_cam_cartesian = my_utils.depth2cam_carte(
            depth=depth_corrected,
            sphere_radius=sphere_radius,
            height=height,
            width=width,
        ) # [H, W, 3]

        # 3. Correct Floor
        if correct_floor:
            pts_cam_cartesian = GeometryTransforms.correct_floor(
                pts_cam_cartesian,
                plot=True
            )
            if verbose:
                print("b. Floor Corrected.")

        # 4. Correct Walls
        if correct_walls:
            if indoor_or_outdoor == 'indoor':
                sky_mask = None
            else:
                sky_mask = GeometryTransforms.get_sky_mask(depth_corrected)
                
            pts_cam_cartesian = GeometryTransforms.correct_walls(
                pts_sph=my_utils.carte2sph_3D(pts_cam_cartesian),
                theta_range=(np.deg2rad(0), np.deg2rad(70)),
                edge=np.deg2rad(15),
                sky_mask=sky_mask
            )
            if verbose:
                print("c. Walls Corrected.")

        #5. remove sky points
        if remove_sky:
            assert indoor_or_outdoor == 'outdoor', "Sky removal can only be done for outdoor scenes."
            sky_mask = GeometryTransforms.get_sky_mask(depth_corrected)
            pts_cam_cartesian[sky_mask] = np.array([np.nan, np.nan, np.nan])
            #TODO: plot a figure showing the sky mask as an overlay over the image.

            if verbose:
                print("d. (Optional) Sky Removed.")

        return pts_cam_cartesian

class SphericalDreamer:

    def __init__(self, pano_depth_temp_dir, pano_width=1440, pano_height=720, seed=119223):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.seed = seed
        self.pano_depth_temp_dir = pano_depth_temp_dir
        self.flux_lora_pano_path = 'checkpoints/pano_lora_720*1440_v1.safetensors'
        self.is_pano_generator_init = False
        self.is_inpainting_model_init = False
        self.is_improve_resolution_model_init = False
        self.is_lama_init = False

    def init_pano_generator(self):
        self.pano_gen_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pano_gen_pipeline.load_lora_weights(self.flux_lora_pano_path) 
        self.pano_gen_pipeline.enable_model_cpu_offload()  
        self.pano_gen_pipeline.enable_vae_tiling()
        
    @torch.no_grad()
    def gen_pano(self, prompt, override_with_inpaint=False, seed_override=None):

        if override_with_inpaint:
            return self.inpaint_pano(
                prompt=prompt,
                pano_rgb=Image.new('RGB', (self.pano_width, self.pano_height), (127,127,127)),
                mask=Image.new('L', (self.pano_width, self.pano_height), 255)
            )

        if not self.is_pano_generator_init:
            self.init_pano_generator()
            self.is_pano_generator_init = True

        seed = self.seed if seed_override is None else seed_override
        pano_rgb = self.pano_gen_pipeline(
            prompt, 
            height=self.pano_height,
            width=self.pano_width,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=50, 
            blend_extend=2,
            guidance_scale=7).images[0]

        # image = image.resize((2048,1024))

        return pano_rgb
    
    @torch.no_grad()
    def estimate_pano_depth(self, pano_rgb):
        """
        args:
            `pano_rgb`: np.array of shape [pano_h,pano_w,3] and values in [0-255]        
        """
        self.depth_estimator = Pano_depth_estimation(
            self.pano_height, 
            self.pano_width, 
            self.pano_depth_temp_dir, 
            self.device, 
            depth_model="DepthAnythingv2"
        )
        pano_depth = self.depth_estimator.get_panodepth(pano_rgb)  #[0-1] 
        return pano_depth  

    def init_inpainting_model(self):

        self.pano_inpaint_pipeline = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        # self.pano_inpaint_pipeline.load_lora_weights(self.flux_lora_pano_path) # Antoine: Do not use the lora for inpainting, it yields worse results. TODO: maybe verify this further
        self.pano_inpaint_pipeline.enable_model_cpu_offload()
        # pipe.enable_vae_tiling() #todo test with or without this?

    def inpaint_pano(self, prompt, pano_rgb, mask, seed_override=None):
        "pano_rgb, mask: PIL.Image"

        if not self.is_inpainting_model_init:
            self.init_inpainting_model()
            self.is_inpainting_model_init = True

        # i. inpainting
        seed = self.seed if seed_override is None else seed_override
        mask = mask.convert("L")
        pano_inpainted_raw = self.pano_inpaint_pipeline(
            prompt=prompt,
            image=pano_rgb,  
            mask_image=mask, 
            strength=1.0,
            height=self.pano_height,
            width=self.pano_width,
            guidance_scale=30.0,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),  
        ).images[0]

        return pano_inpainted_raw

    def blend(self, pano_rgb, pano_inpainted_raw, missing_info_mask, horizon_mask):

        #ii. compose blending
        mask_blend1 = missing_info_mask
        pano_blend1 = self._blend(
            pano_inpainted_raw, 
            pano_rgb, 
            mask_blend1, 
            mode='compose'
        )

        # iii. seamless blending
        mask_blend2=horizon_mask
        pano_blend2 = self._blend(
            pano_inpainted_raw, 
            pano_blend1, 
            mask_blend2,
            mode='seamless'
        )

        return pano_blend1, pano_blend2, mask_blend1, mask_blend2 #TODO: only pano_blend1 is needed

    def _blend(self, src, dst, mask, mode):
        "Blends two images together, guided by mask. All arguments should be PIL.Image"

        # Naive blending. Just compose the images
        if mode == 'compose':
            pano_blended = Image.composite(src, dst, mask)

        # Seamless blending, with smoothing along the mask edges
        elif mode == 'seamless':
            pano_blended = my_utils.seamless_blend(src, dst, mask)
        else:
            raise ValueError(f"Unknown blending mode: {mode}. Mode should either be 'seamless' or 'compose'.")

        return pano_blended

    def init_improve_resolution_model(self):

        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        )
        self.improve_resolution_pipeline = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
        # self.improve_resolution_pipeline.load_lora_weights(self.flux_lora_pano_path)  # change this.
        self.improve_resolution_pipeline.enable_model_cpu_offload()

    def improve_pano_resolution(self, pano_rgb, prompt, controlnet_conditioning_scale=0.2):

        if not self.is_improve_resolution_model_init:
            self.init_improve_resolution_model()
            self.is_improve_resolution_model_init = True

        image = self.improve_resolution_pipeline(
            prompt=prompts, 
            control_image=pano_rgb,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=50, 
            guidance_scale=3.5,
            height=pano_rgb.size[1],
            width=pano_rgb.size[0],
            generator=torch.Generator("cpu").manual_seed(self.seed) 
        ).images[0]
        return image
    
    def init_lama(self):
        from src.lama import LamaInpainting
        self.lama_model = LamaInpainting()

    def lama_inpaint(self, image:Image, mask:Image):
        """
        image: PIL.Image (RGB)
        mask: PIL.Image (L)
        """
        if not self.is_lama_init:
            self.init_lama()
            self.is_lama_init = True

        return Image.fromarray(self.lama_model(image, mask))

def camera_translation(pose, translation):
    """
    pose: np.array of shape [4,4]
    translation: np.array of shape [3,] in world coordinates
    """
    pose2 = pose.copy()
    pose2[:3, 3] += translation
    return pose2

def load_rgbd_pano(dream, save_dir_, override_depth_with_ones=False):

    pano_rgb = Image.open(f"{save_dir_}/dream_{dream:02d}/XX_pano_rgb.png")
    depth = np.load(f"{save_dir_}/dream_{dream:02d}/XX_depth.npy")
    if override_depth_with_ones:
        depth = np.ones_like(depth)  
        print("WARNING: depth override to ones for debugging purposes")
    colors = np.array(pano_rgb)/255.0
    return colors, depth

def render(all_pts_world, all_colors_world, pose):

    # convert to ERP + Depth representation
    points_3D_cam2_sph = my_utils.world2cam_sph_3D(all_pts_world, pose)  
    depth_cam2 = points_3D_cam2_sph[..., 2] # [N,] 
    points_2D_cam2_sph = points_3D_cam2_sph[..., :2]
    points_2D_cam2_erp = my_utils.sph2erp_2D(points_2D_cam2_sph, height, width)  # [N, 2]
    # Splatting +  Interpolation
    warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels = my_utils.splatting_and_interpolation(
        colors=all_colors_world,
        depth_cam2=depth_cam2,
        coord_cam2=points_2D_cam2_erp,
        height=height,
        width=width,
        interpolation_mode='rounded',
    )
    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels

def get_missing_info_mask(operations, visited_pixels, log_mask=True):
    missing_info_masks = [~visited_pixels]
    for op in operations:
        missing_info_masks.append(op(missing_info_masks[-1]))
    if log_mask:
        missing_info_masks_tile = my_utils.tile_image([my_utils.numpy_to_PIL(m) for m in missing_info_masks])
        missing_info_masks_tile.save(f"{save_dir_}/align_{i:02d}/02_missing_info_masks_tile.png")
    missing_info_mask = missing_info_masks[-1]
    return missing_info_mask

def check_partition(*masks):
    """Return True if masks are disjoint and cover the full image."""
    # disjointness
    total = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        if np.any(total & m):
            return False
        total |= m
    # full coverage
    return np.all(total)

def get_harmonic_blending_mask(missing_info_mask):
    """
    missing_info_mask: np.array of shape [H, W] with dtype bool. True where info is missing i.e. where we inpainted
    """
    mask1 = ~missing_info_mask
    mask2 = missing_info_mask
    boundary = find_boundaries(mask1, mode='inner', background=False)  # [H, W]
    mask1 = mask1 & (~boundary)
    mask2 = mask2 & (~boundary)
    assert check_partition(mask1, mask2, boundary), "Masks are not a valid partition of the image"
    return mask1, mask2, boundary

def get_mask_fixed(forward, pts):
    # point should be on cartesian coordinates in camera frame
    cosine_similarity = pts @ forward / (np.linalg.norm(forward) * np.linalg.norm(pts, axis=-1) + 1e-8)
    mask_fixed = cosine_similarity >= 0
    return mask_fixed

def harmonic_blend_of_depths(colors, warped_depth_interp, depth_estimated, missing_info_mask, pose, sphere_radius, height, width, logging=True):
    """ Inputs are in HxW format except colors which is HxWx3 
    Given the two depth map (interpolated and estimated), it merges with the following constraints:
        - points in the good region of warped_depth_interp stay unchanged
        - points in the missing region of warped_depth_interp are moved as little as possible to make it both continious and close to depth_estimated
    Returns:
        - pts2_deformed: np.array of shape [N, 3] in world coordinates of the points coming from depth_estimated, withing the inpainted region, after harmonic deformation
        - colors2: np.array of shape [N, 3] with values in [0-1] corresponding to pts2_deformed
        - pcd_harmonic: PointCloud object with the full blended pointcloud (More points than pts2_deformed, repetition with existing points)
        - blended_depth_harmonic: np.array of shape [H, W] with the blended depth
    """

    def _log_masks(mask1, mask2, mask_boundary):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(mask1, cmap='gray')
        plt.title("Mask 1 (good points)")
        plt.subplot(1,3,2)
        plt.imshow(mask2, cmap='gray')
        plt.title("Mask 2 (to be deformed)")
        plt.subplot(1,3,3)
        plt.imshow(mask_boundary, cmap='gray')
        plt.title("Mask boundary")
        plt.savefig(f"{save_dir_}/align_{i:02d}/07_harmonic_blending_masks.png")
        plt.show()
    
    mask_keep, mask_deform, mask_boundary = get_harmonic_blending_mask(missing_info_mask)

    all_pts_keep = my_utils.depth2world(
        depth=warped_depth_interp, pose=pose, sphere_radius=sphere_radius, height=height, width=width
    ) # here camera pose is not good maybe ??
    all_pts_deform = my_utils.depth2world(
        depth=depth_estimated, pose=pose, sphere_radius=sphere_radius, height=height, width=width
    )
    pts_keep = all_pts_keep[mask_keep] # these are already good
    pts_target_boundary = all_pts_keep[mask_boundary] 
    pts_deform_exb = all_pts_deform[mask_deform] # these need to be deformed by mooving the boundary points to the target boundary points
    pts_deform_boundary = all_pts_deform[mask_boundary]
    pts_deform = np.concatenate((pts_deform_exb, pts_deform_boundary), axis=0)
    _mask_boundary = np.concatenate((np.zeros(pts_deform_exb.shape[0], dtype=bool), np.ones(pts_deform_boundary.shape[0], dtype=bool)), axis=0)
    # mask_fixed = get_mask_fixed(translation_direction, my_utils.world2cam_carte_3D(pts_deform, pose))
    # verify_mask_fixed(translation_direction, pano_rgb_inpainted)

    # Deformation
    assert np.any(np.isnan(pts_deform)) == False, "Error: pts_deform contains NaNs"
    assert np.any(np.isnan(pts_target_boundary)) == False, "Error: pts_target_boundary contains NaNs"
    t0 = time.time()
    pts_deformed, _ = my_utils.harmonic_deform_pipeline(
        P=pts_deform,
        mask_fixed=np.zeros(pts_deform.shape[0], dtype=bool),
        mask_boundary=_mask_boundary,
        target_boundary=pts_target_boundary,
        n_coarse=10000,
        every=5,
        max_fixed=2000,
        k=10, m=3
    )
    t1 = time.time()
    print(f"Harmonic deformation took {t1 - t0:.1f}s")

    pts_deformed_exb, pts_deformed_boundary = np.split(pts_deformed, [pts_deform_exb.shape[0]], axis=0)
    pts_deformed = np.concatenate((pts_deformed_exb, pts_deformed_boundary), axis=0)
    colors2_exb = colors[mask_deform]
    colors2_boundary = colors[mask_boundary]
    colors2 = np.concatenate((colors2_exb, colors2_boundary), axis=0)

    # Visualization & pointcloud
    if logging:
        _log_masks(mask_keep, mask_deform, mask_boundary)
        # TODO: What does the new spherical image looks like from pose ? With deformed points ?

        # visualize blended depth and pointcloud from current camera
        pts_3D_carte_new = np.zeros((height, width, 3), dtype=np.float32)
        pts_3D_carte_new[mask_keep] = pts_keep
        pts_3D_carte_new[mask_deform] = pts_deformed_exb
        pts_3D_carte_new[mask_boundary] = pts_deformed_boundary
        blended_depth_harmonic = my_utils.world2cam_sph_3D(pts_3D_carte_new, pose)[..., 2]
        pcd_harmonic = PointCloud(
            pts=pts_3D_carte_new,
            colors=colors
        )

        plt.figure()
        plt.imshow(blended_depth_harmonic, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Harmonic')
        plt.savefig(f"{save_dir_}/align_{i:02d}/08_blended_depth_harmonic.png")
        plt.show()

        return pts_deformed, colors2, pcd_harmonic, blended_depth_harmonic

    return pts_deformed, colors2

def naive_blend_of_depths(colors, warped_depth_interp, depth_estimated, missing_info_mask, pose, sphere_radius, height, width, logging=True):

    if logging:

        blended_depth = np.zeros_like(warped_depth_interp)
        blended_depth[missing_info_mask] = depth_estimated[missing_info_mask]
        blended_depth[~missing_info_mask] = warped_depth_interp[~missing_info_mask]

        pcd_naive = PointCloud(
            pts=my_utils.depth2world(
                depth=blended_depth, pose=pose, sphere_radius=sphere_radius, height=height, width=width
            ),
            colors=colors
        )

        plt.figure()
        plt.imshow(blended_depth, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Naive')
        plt.savefig(f"{save_dir_}/align_{i:02d}/08_blended_depth_naive.png")
        plt.show()

    return pcd_naive, blended_depth


if __name__ == "__main__":
    # ---- args ----
    debug = False
    skip_phase1 = True
    skip_inpainting = False
    save_dir = "OUTPUTS/SphericalDreamerRecurse"

    # dreaming args
    num_dreams = 5
    seeds = [119224, 119224+9, 119224+20, 119224+33, 119224+45]
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    sphere_radius = 1.0
    delta_walk = FAR 
    override_with_inpaint=False
    width = 1440
    height = 720
    prompts = [
        "A realistic illustration of a college campus. In the middle ground, several academic buildings with brick facades and large windows stand prominently. In the background, a bright blue sky with scattered clouds stretches across the scene. In the foreground, a few elements commonly found on campus, such as students walking, bicycles parked along a path, and a grassy lawn with trees, add depth and life to the scene.",
        "A wide panoramic landscape with a bright blue sky, majestic mountains in the background, a calm turquoise sea in the foreground, and lush greenery along the shore. The scene should feel vibrant, sunny, and relaxing, like a holiday postcard photograph, with realistic lighting and high detail.",
        "A serene forest scene with a small stream, dappled sunlight filtering through the leaves, realism style.",
        "A bustling city street at night, neon lights reflecting on wet pavement, realism style.",
        # "Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.",
        # "A wide field under daylight, covered in lush green grass with worn paths where the grass has been trampled by many footsteps. In the center of the field stands a large concert stage, decorated with bold triangular patterns. On the stage rests a single guitar, but no performers are present. In front of the stage, a lively crowd gathers, waiting for the show to begin."
    ]
    expnames=[
        "24_campus",
        "24_seaside",
        "24_forest",
        "24_city",
        # "09_bali_aligned", 
        # "forest", 
        # "city", 
        # "beach", 
        # "the_stage",
    ]
    indoor_or_outdoor_list = [
        'outdoor',
        'outdoor',
        'indoor',
        'outdoor',
        # 'outdoor',
    ]
    # ---------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, help='Experiment ID to run (0-4)', default=3)
    if debug:
        args = parser.parse_args([
            '--exp_id', '0'
        ])
        for _ in range(10):
            print(f"/!\ DEBUG MODE IS ON. Running exp {args.exp_id}/!\ ")
    else:
        args = parser.parse_args()

    expname, prompt, indoor_or_outdoor = expnames[args.exp_id], prompts[args.exp_id], indoor_or_outdoor_list[args.exp_id]

    # 0. Initialization
    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp'
    )

    save_dir_ = f"{save_dir}/{expname}"
    
    # PHASE 0. GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH
    if not skip_phase1:
        for i in range(num_dreams):
            print(f"--- Dreaming Phase {i:02d} / {num_dreams} ---")

            # Generate panorama & Estimate Depth
            pano_rgb = spherical_dreamer.gen_pano(prompt=prompt, override_with_inpaint=override_with_inpaint, seed_override=seeds[i])
            depth = spherical_dreamer.estimate_pano_depth(pano_rgb=np.array(pano_rgb))
            os.makedirs(os.path.join(save_dir_, f"dream_{i:02d}"), exist_ok=True)
            pano_rgb.save(f"{save_dir_}/dream_{i:02d}/XX_pano_rgb.png")
            np.save(f"{save_dir_}/dream_{i:02d}/XX_depth.npy", depth)
            my_utils.depth_numpy_to_PIL(depth).save(f"{save_dir_}/dream_{i:02d}/XX_depth.png")
            my_utils.depth_numpy_to_figure(depth).savefig(f"{save_dir_}/dream_{i:02d}/XX_depth_figure.png")


    pointclouds = {}
    all_pts_world = np.array([]).reshape(0, 3)
    all_colors_world = np.array([]).reshape(0, 3)

    depth_threshold = 0.9
    depth_metric_threshold = GeometryTransforms.depth_transform(
        np.array([depth_threshold]),
        method="inv",
        n=NEAR,
        f=FAR,
        gamma=5
    )[0]
    opening_kwargs = {
        # 'cut_distance_percentile': 95,
        'cut_distance': depth_metric_threshold,
    }
    

    # PHASE I. INIT FIRST SPHERE
    print(f"--- Opening first sphere ---")
    pose1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)    


    colors1, depth1 = load_rgbd_pano(
        dream=0,
        save_dir_=save_dir_
    )
    pts1_carte_corrected = GeometryTransforms.run_corrective_pipeline(
        depth=depth1,
        sphere_radius=sphere_radius,
        correct_floor=False,
        correct_walls=False,
        remove_sky=False,
        indoor_or_outdoor=indoor_or_outdoor,
        verbose=True
    )
    sphere1 = Sphere(
        pose1, pts1_carte_corrected, colors1, 
        forward_carte=translation_direction,
        opening_kwargs=opening_kwargs,
    )

    # PHASE II. ALIGNMENT PHASE WITH INPAINTING + HARMONIC BLENDING
    for i in range(1, num_dreams):
        print(f"--- Inpainting+Alignment Phase {i:02d} / {num_dreams-1} ---")
        os.makedirs(os.path.join(save_dir_, f"align_{i:02d}"), exist_ok=True)

        # 1. Load new sphere and open it (left)
        colors2, depth2 = load_rgbd_pano(
            dream=i,
            save_dir_=save_dir_
        )
        pts2_carte_corrected = GeometryTransforms.run_corrective_pipeline(
            depth=depth2,
            sphere_radius=sphere_radius,
            correct_floor=False,
            correct_walls=False,
            remove_sky=False,
            indoor_or_outdoor=indoor_or_outdoor,
            verbose=True
        )
        
        sphere2 = Sphere(
            None, pts2_carte_corrected, colors2, 
            forward_carte=translation_direction,
            opening_kwargs=opening_kwargs,
        )

        # 2. Move camera
        delta_walk =  2.0 * depth_metric_threshold   
        pose2 = camera_translation(pose1, delta_walk * translation_direction)
        sphere2.update_pose(pose2)
        
        print('Loaded and opened sphere!')

        # 4. Go to intermediate camera (between cam1 and cam2)
        pose_intermediate = camera_translation(pose2, -delta_walk/2 * translation_direction)
        pose_intermediate_bis = camera_translation(pose1, delta_walk/2 * translation_direction)
        # TODO: add some verticality here, i.e., not only along translation_direction
        assert np.allclose(pose_intermediate, pose_intermediate_bis), "Error in camera intermediate pose computation"

        # 5. Render points from sphere2 (opened right) + sphere2 (opened left), from the intermediate camera
        warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels = render(
            all_pts_world=np.concatenate((
                sphere1.right_opened.get_world_pcd().pts, sphere2.left_opened.get_world_pcd().pts
            ), axis=0), 
            all_colors_world=np.concatenate((
                sphere1.right_opened.get_world_pcd().colors, sphere2.left_opened.get_world_pcd().colors
            ), axis=0), 
            pose=pose_intermediate
        )
        print("Rendered all points from intermediate camera!")

        # 6. Get missing info mask
        operations = [
            partial(minimum_filter, size=(3,3), axes=(0,1)),
            partial(maximum_filter, size=(3,3), axes=(0,1)),
            partial(maximum_filter, size=(3,3), axes=(0,1)),
            partial(maximum_filter, size=(3,3), axes=(0,1)),
            # partial(maximum_filter, size=(8, 8), axes=(0,1)),
        ]
        missing_info_mask = get_missing_info_mask(operations, visited_pixels, log_mask=True) 
        where_depth_nan = np.isnan(warped_depth_interp)
        missing_info_mask = missing_info_mask | where_depth_nan
        inpainting_mask = missing_info_mask # TODO: (Antoine, 14 oct) The inpainting mask is currently composed of both <<large missing regions due to limited covering of the main spheres>> and <<small holes due to occlusions>>. We could separate these two cases and do something neater?.

        warped_img_interp[missing_info_mask] = np.nan
        warped_depth_interp[missing_info_mask] = np.nan
        my_utils.numpy_to_PIL(warped_img).save(f"{save_dir_}/align_{i:02d}/01_warped_img.png")
        my_utils.depth_numpy_to_PIL(warped_depth).save(f"{save_dir_}/align_{i:02d}/01_warped_depth.png")    
        my_utils.numpy_to_PIL(warped_img_interp).save(f"{save_dir_}/align_{i:02d}/03_warped_img_interp.png")
        my_utils.depth_numpy_to_PIL(warped_depth_interp).save(f"{save_dir_}/align_{i:02d}/03_warped_depth_interp.png")
        my_utils.depth_numpy_to_figure(warped_depth_interp).savefig(f"{save_dir_}/align_{i:02d}/03_warped_depth_interp_figure.png")
        # np.save(f"{save_dir_}/align_{i:02d}/03_warped_depth_interp.npy", warped_depth_interp)
        
        # 7. Inpainting
        overlay_before = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img_interp, inpainting_mask, alpha=0.5)) 
        overlay_before.save(f"{save_dir_}/align_{i:02d}/04_overlay_before_inpainting.png")
        if not skip_inpainting: 
            pano_inpainted_raw = spherical_dreamer.inpaint_pano(
                prompt=prompt, 
                pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
                mask=my_utils.numpy_to_PIL(inpainting_mask)
            )
            pano_inpainted_raw.save(f"{save_dir_}/align_{i:02d}/XX_pano_rgb_inpainted_raw.png")
        else:
            pano_inpainted_raw = Image.open(f"{save_dir_}/align_{i:02d}/XX_pano_rgb_inpainted_raw.png")
        pano_inpainted_raw.save(f"{save_dir_}/align_{i:02d}/04_pano_rgb_inpainted_raw.png")

        # 7. Inpainting seamless blending
        pano_blend1, pano_blend2, mask_blend1, mask_blend2 = spherical_dreamer.blend(
            pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
            pano_inpainted_raw=pano_inpainted_raw,
            missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
            horizon_mask=my_utils.numpy_to_PIL(np.zeros_like(missing_info_mask).astype('bool')),
        ) 
        #TODO: since we removed horizon, check the blending strategy again. It is `compose` everywhere now. Should be seamless for the large inapainted part
        #TODO: Also, Check if we need both blend1 and blend2
        
        mask_blend1.save(f"{save_dir_}/align_{i:02d}/05_blend1_mask.png")
        mask_blend2.save(f"{save_dir_}/align_{i:02d}/05_blend2_mask.png")
        pano_blend1.save(f"{save_dir_}/align_{i:02d}/05_blend1_pano_rgb_inpainted.png")
        pano_blend2.save(f"{save_dir_}/align_{i:02d}/05_blend2_pano_rgb_inpainted.png")

        pano_rgb_inpainted = pano_blend2
        pano_rgb_inpainted.save(f"{save_dir_}/align_{i:02d}/06_pano_rgb_inpainted.png") #TODO: this is the same as blend2. Remove repetition

        # 8. Estimate depth
        # TODO: (Antoine, 16 OCT) LayerPANO3D Has a depth inpainting model, which may be better than this + harmonic blending. Worth testing.)
        if not skip_inpainting:
            depth_estimated = spherical_dreamer.estimate_pano_depth(
                pano_rgb=np.array(pano_rgb_inpainted)
            )
            np.save(f"{save_dir_}/align_{i:02d}/XX_estimated_depth.npy", depth_estimated)
        else:
            depth_estimated = np.load(f"{save_dir_}/align_{i:02d}/XX_estimated_depth.npy")
        # depth_estimated=np.ones_like(depth_estimated) * sphere_radius  
        # print("WARNING: estimated depth override to ones for debugging purposes")
        my_utils.depth_numpy_to_PIL(depth_estimated).save(f"{save_dir_}/align_{i:02d}/07_estimated_depth.png")
        my_utils.depth_numpy_to_figure(depth_estimated).savefig(f"{save_dir_}/align_{i:02d}/07_estimated_depth_figure.png")


        # 9. Blend depth
        new_colors = (np.array(pano_rgb_inpainted)/255.0)

        # (Naive blending)
        # TODO: (Antoine): I think the variable below should be inpainting_mask instead of missing_info_mask
        pcd_naive, blended_depth_naive = naive_blend_of_depths(
            colors=new_colors,
            warped_depth_interp=warped_depth_interp,
            depth_estimated=depth_estimated,
            missing_info_mask=missing_info_mask,
            pose=pose_intermediate,
            sphere_radius=sphere_radius,
            height=height,
            width=width,
            logging=True
        )

        # (Harmonic blending)
        pts_deformed_world, new_colors, pcd_harmonic, blended_depth_harmonic = harmonic_blend_of_depths(
            colors=new_colors,
            warped_depth_interp=warped_depth_interp,
            depth_estimated=depth_estimated,
            missing_info_mask=missing_info_mask,
            pose=pose_intermediate,
            sphere_radius=sphere_radius,
            height=height,
            width=width,
            logging=True
        )
        pointclouds[f"inpaint_{i:02d}"] = {}
        pointclouds[f"inpaint_{i:02d}"]['blended_naive_w_excess'] = pcd_naive
        pointclouds[f"inpaint_{i:02d}"]['blended_harmonic_w_excess'] = pcd_harmonic
        pointclouds[f"inpaint_{i:02d}"]["blended_harmonic"] = PointCloud(
            pts=pts_deformed_world,
            colors=new_colors
        )

        def is_point_in_camera_forward_space(point_positions,
                                            camera_position,
                                            forward_vector,
                                            tolerance=1e-12):
            """
            Determine whether one or more 3D points lie in the half-space
            in front of the plane orthogonal to `forward_vector`
            passing through `camera_position`.

            Parameters
            ----------
            point_positions : array-like, shape (..., 3)
                One or more 3D points. Supports arbitrary leading batch dimensions.
            camera_position : array-like, shape (3,)
                The 3D location of the camera.
            forward_vector : array-like, shape (3,)
                The camera's forward direction vector (does not need to be normalized).
            tolerance : float, optional
                Numerical tolerance for deciding whether a point on the plane counts as "in front".

            Returns
            -------
            np.ndarray of bool
                Boolean array of shape (...) — True for points in the camera’s forward half-space,
                False for points behind it.
            """

            # Convert to arrays
            point_positions = np.asarray(point_positions, dtype=float)
            camera_position = np.asarray(camera_position, dtype=float)
            forward_vector = np.asarray(forward_vector, dtype=float)

            # Check that the forward vector is valid
            if np.allclose(forward_vector, 0):
                raise ValueError("forward_vector must be a non-zero vector.")

            # Vector(s) from camera to point(s) – broadcasting works automatically
            vectors_camera_to_points = point_positions - camera_position

            # Signed distance(s) along the forward direction
            signed_distances = np.sum(vectors_camera_to_points * forward_vector, axis=-1)

            # True if in or beyond the forward half-space
            return signed_distances >= -tolerance

        def split_new_points(pts, colors, pose1, pose2, forward):
            # (Antoine, 16 Oct) This function will pose problems if we want to do anything different than a straight line path.
            """
            Split points between points belonging to sphere1, sphere2, and neutral points.
            Points are distrbuted as follows:
                - pts on the left side of cam1 belongs to sphere 1
                - pts on the right side of cam2 belongs to sphere 2
                - pts in between are neutral points
            """
            cam_loc_1 = pose1[:3, 3]
            cam_loc_2 = pose2[:3, 3]
            where_sphere1 = is_point_in_camera_forward_space(pts, cam_loc_1, -forward)  # left of cam1
            where_sphere2 = is_point_in_camera_forward_space(pts, cam_loc_2, forward)   # right of cam2
            where_neutral = ~(where_sphere1 | where_sphere2)
            pts1, colors1 = pts[where_sphere1], colors[where_sphere1]
            pts2, colors2 = pts[where_sphere2], colors[where_sphere2]
            pts_neutral, colors_neutral = pts[where_neutral], colors[where_neutral]
            return (pts1, colors1), (pts2, colors2), (pts_neutral, colors_neutral)

        # Add new points to their corresponding spheres.
        (new_pts1, new_colors1), (new_pts2, new_colors2), (new_pts_neutral, new_colors_neutral) = split_new_points(
            pts_deformed_world, new_colors, pose1, pose2, translation_direction
        )
        sphere1.add_new_points(my_utils.world2cam_sph_3D(new_pts1, pose1), new_colors1)
        sphere2.add_new_points(my_utils.world2cam_sph_3D(new_pts2, pose2), new_colors2)

        # 10. Add all new points to world points, including inpainted+deformed points and points from the current dream.
        pointclouds[f'dream_{i:02d}'] = {}
        pointclouds[f"dream_{i:02d}"]['sphere1_init'] = sphere1.closed.get_world_pcd()
        pointclouds[f"dream_{i:02d}"]['sphere2_init'] = sphere2.closed.get_world_pcd()
        #10.a Points from sphere1
        if i == 1: # first iteration: sphere1 only has right opened
            pointclouds[f"dream_{i:02d}"]['sphere1_open'] = sphere1.right_opened.get_world_pcd()
            all_pts_world = np.concatenate((all_pts_world, sphere1.right_opened.get_world_pcd().pts), axis=0)
            all_colors_world = np.concatenate((all_colors_world, sphere1.right_opened.get_world_pcd().colors), axis=0)
        else: # later iterations: sphere1 has both opened
            pointclouds[f"dream_{i:02d}"]['sphere1_open'] = sphere1.both_opened.get_world_pcd()
            all_pts_world = np.concatenate((all_pts_world, sphere1.both_opened.get_world_pcd().pts), axis=0)
            all_colors_world = np.concatenate((all_colors_world, sphere1.both_opened.get_world_pcd().colors), axis=0)
        #10.b Neutral points
        all_pts_world = np.concatenate((all_pts_world, new_pts_neutral), axis=0)
        all_colors_world = np.concatenate((all_colors_world, new_colors_neutral), axis=0)
        #10.c Points from sphere2 (only last iter)
        if i == num_dreams - 1: 
            pointclouds[f"dream_{i:02d}"]['sphere2_open'] = sphere2.left_opened.get_world_pcd()
            all_pts_world = np.concatenate((all_pts_world, sphere2.left_opened.get_world_pcd().pts), axis=0)
            all_colors_world = np.concatenate((all_colors_world, sphere2.left_opened.get_world_pcd().colors), axis=0)

        # 11. Log final pointcloud
        pointclouds[f"dream_{i:02d}"][f"total"] = PointCloud(
            pts=all_pts_world,
            colors=all_colors_world
        )

        # 12. Adjust sphere1 to be sphere2 for next iteration
        sphere1 = sphere2
        pose1 = pose2

        # save pcd
        with open(f"{save_dir_}/pointclouds.pkl", 'wb') as f:
            pkl.dump(pointclouds, f)

    print("PYTHON SCRIPT SUCCESSFULLY RUN TO THE END !")