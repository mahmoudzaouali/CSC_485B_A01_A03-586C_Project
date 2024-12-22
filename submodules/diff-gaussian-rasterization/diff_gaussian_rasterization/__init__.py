#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

# def cpu_deep_copy_tuple(input_tuple):
#     copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
#     return tuple(copied_tensors)


def cpu_deep_copy_tuple(input_tuple):
    keys = [
        "bg", "means3D", "radii", "colors_precomp", "scales", "rotations",
        "scale_modifier", "cov3Ds_precomp", "viewmatrix", "projmatrix",
        "tanfovx", "tanfovy", "grad_out_color", "sh", "sh_degree", "campos",
        "geomBuffer", "num_rendered", "binningBuffer", "imgBuffer", "debug"
    ]
    copied_tensors = {key: item.cpu().clone() if isinstance(item, torch.Tensor) else item for key, item in zip(keys, input_tuple)}
    return copied_tensors

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:

            # print(f"colors_precomp: {colors_precomp.shape}, dtype: {colors_precomp.dtype}")

            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)



        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered

        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, sh, radii, *geomBuffer, *binningBuffer, *imgBuffer)  # Save tensors for backward



        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings

        # Retrieve all saved tensors
        (
            colors_precomp, means3D, scales, rotations, cov3Ds_precomp, sh, radii,
            *saved_buffers  # Remainder tensors are unpacked into saved_buffers
        ) = ctx.saved_tensors      


        num_views = len(saved_buffers) // 3  # Assuming 3 lists: geomBuffer, binningBuffer, imgBuffer

        # Split saved_buffers into individual lists
        geomBuffer = saved_buffers[:num_views]        # First num_views tensors
        binningBuffer = saved_buffers[num_views:2*num_views]  # Next num_views tensors
        imgBuffer = saved_buffers[2*num_views:]       # Last num_views tensors



        # # Example: Print to verify the reconstructed lists
        # print("Number of views:", num_views)
        # print("GeomBuffer sizes:", [gb.size() for gb in geomBuffer])
        # print("BinningBuffer sizes:", [bb.size() for bb in binningBuffer])
        # print("ImgBuffer sizes:", [ib.size() for ib in imgBuffer])

        num_rendered = num_rendered.squeeze(-1)  # Remove singleton dimensions

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii,  # Batched radii
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, # Batched view matrices
                raster_settings.projmatrix, # Batched projection matrices
                raster_settings.tanfovx,  # Batched tan(FoVx)
                raster_settings.tanfovy,  # Batched tan(FoVy)
                grad_out_color,  # Batched gradient
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos, # Batched camera positions
                geomBuffer,     # Batched geometry buffers
                num_rendered,   # Tensor/list with per-view values
                binningBuffer, # Batched binning buffers
                imgBuffer, # Batched image buffers
                raster_settings.debug)
        
        

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             

             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: torch.Tensor # Batched heights: Tensor of shape (B,)
    image_width: torch.Tensor  # Batched widths: Tensor of shape (B,)
    tanfovx : torch.Tensor     # Batched FOVx: Tensor of shape (B,) in float 
    tanfovy : torch.Tensor     # Batched FOVy: Tensor of shape (B,) in float 
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor # Batched view matrices: Tensor of shape (B, 4, 4)
    projmatrix : torch.Tensor # Batched projection matrices: Tensor of shape (B, 4, 4)
    sh_degree : int           # Batched camera positions: Tensor of shape (B, 3)  
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix, # Batched view matrices
                raster_settings.projmatrix) # Batched projection matrices
            
        return visible

    def forward(self, 
                means3D, 
                means2D, 
                opacities, 
                shs = None, 
                colors_precomp = None, 
                scales = None, 
                rotations = None, 
                cov3D_precomp = None):
        
        """
        Perform batched rasterization by delegating to the CUDA backend.
        """

        raster_settings = self.raster_settings
        
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

