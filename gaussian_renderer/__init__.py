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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Handle single Camera objects by wrapping them in a list
    if not isinstance(viewpoint_camera, (list, tuple)):
        viewpoint_camera = [viewpoint_camera]

    # Extract batched camera parameters
    batch_size = len(viewpoint_camera)
    image_heights = torch.tensor([cam.image_height for cam in viewpoint_camera], dtype=torch.int32, device="cuda")    
    image_widths = torch.tensor([cam.image_width for cam in viewpoint_camera], dtype=torch.int32, device="cuda")
    tanfovx = torch.tensor([math.tan(cam.FoVx * 0.5) for cam in viewpoint_camera], dtype=torch.float32)
    tanfovy = torch.tensor([math.tan(cam.FoVy * 0.5) for cam in viewpoint_camera], dtype=torch.float32)

    world_view_transforms = torch.stack([cam.world_view_transform.clone().detach() for cam in viewpoint_camera], dim=0).to("cuda")  # (B, 4, 4)
    full_proj_transforms = torch.stack([cam.full_proj_transform.clone().detach() for cam in viewpoint_camera], dim=0).to("cuda")  # (B, 4, 4)    
    camera_centers = torch.stack([cam.camera_center.clone().detach() for cam in viewpoint_camera], dim=0).to("cuda")  # (B, 3)


    # world_view_transforms = torch.stack([cam.world_view_transform for cam in viewpoint_camera], dim=0) # (B, 4, 4) not sure about this size
    # full_proj_transforms = torch.stack([cam.full_proj_transform for cam in viewpoint_camera], dim=0) # (B, 4, 4) not sure about this size
    # camera_centers = torch.stack([cam.camera_center for cam in viewpoint_camera], dim=0) # (B, 3) 
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=image_heights, # Tensor of shape (B,)
        image_width=image_widths, # Tensor of shape (B,)
        tanfovx=tanfovx, # Tensor of shape (B,)
        tanfovy=tanfovy, # Tensor of shape (B,)
        bg=bg_color, # Background tensor (batched or single shared tensor)
        scale_modifier=scaling_modifier, # Single scalar
        viewmatrix=world_view_transforms, # Tensor of shape (B, 4, 4)
        projmatrix=full_proj_transforms, # Tensor of shape (B, 4, 3)
        sh_degree=pc.active_sh_degree, # Single scalar (SH degree)
        campos= camera_centers,  # Tensor of shape (B, 3)
        prefiltered=False, # Boolean
        debug=pipe.debug # Boolean
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            
            # Compute direction vectors for batched cameras
            dir_pp = pc.get_xyz.unsqueeze(0) - camera_centers.unsqueeze(1) # Shape: (B, P, 3
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) 
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=2, keepdim=True) # Shape: (B, P, 3)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_images, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Perform the comparison for each tensor in the list
    visibility_filter = (radii > 0).to(torch.bool)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # All the results are batched
    return {"render": rendered_images, 
            "viewspace_points": screenspace_points, 
            "visibility_filter" : visibility_filter, 
            "radii": radii} 
