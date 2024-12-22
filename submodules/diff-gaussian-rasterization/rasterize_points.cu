/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <unordered_map>

// This map is used  to store events for each view
static std::unordered_map<int, cudaEvent_t> forwardEvents;

// We use a map with shared pointers to store CUDAStream objects of the forward pass to store them to be used for the backward pass
static std::unordered_map<int, std::shared_ptr<at::cuda::CUDAStream>> forwardStreams;


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<torch::Tensor, 
torch::Tensor, 
torch::Tensor, 
std::vector<torch::Tensor>, 
std::vector<torch::Tensor>, 
std::vector<torch::Tensor>>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix, // Batched view matrices
	const torch::Tensor& projmatrix, // Batched projection matrices
	const torch::Tensor& tan_fovx, 	 // Batched tan(FoVx). As we are using the same cameras, a common parameter could have been passed 
	const torch::Tensor& tan_fovy,	 // Batched tan(FoVy). Same thing for this parameter with FoVx
    const torch::Tensor& image_height,	// Batched image heights. We are using the same image size for all views. Only one value could have been used here
    const torch::Tensor& image_width,	// Batched image widths. We are using the same image size for all view. Only one values could have been used.
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  if (viewmatrix.ndimension() != 3 || viewmatrix.size(1) != 4 || viewmatrix.size(2) != 4) {
	AT_ERROR("viewmatrices must have dimensions (num_views, 4, 4)");
  }

  if (projmatrix.ndimension() != 3 || projmatrix.size(1) != 4 || projmatrix.size(2) != 4) {
        AT_ERROR("projmatrices must have dimensions (num_views, 4, 4)");
  }
  if (campos.ndimension() != 2 || campos.size(1) != 3) {
	AT_ERROR("campos must have dimensions (num_views, 3)");
  }

  const int num_views = viewmatrix.size(0);  // We find the Batch size passed from python to use it here
 

  const int P = means3D.size(0);
  int H = image_height[0].item<int>();
  int W = image_width[0].item<int>();


  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  // Preallocate results as vectors
  std::vector<torch::Tensor> rendered_counts(num_views, torch::zeros({1}, int_opts));
  std::vector<torch::Tensor> out_colors(num_views, torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts));
  std::vector<torch::Tensor> radii_buffers(num_views, torch::full({P}, 0, int_opts));
  std::vector<torch::Tensor> geomBuffers(num_views, torch::empty({0}, torch::TensorOptions().dtype(torch::kByte).device(means3D.device())));
  std::vector<torch::Tensor> binningBuffers(num_views, torch::empty({0}, torch::TensorOptions().dtype(torch::kByte).device(means3D.device())));
  std::vector<torch::Tensor> imgBuffers(num_views, torch::empty({0}, torch::TensorOptions().dtype(torch::kByte).device(means3D.device())));


  // Process each view
  for (int v = 0; v < num_views; ++v) {

	float tan_fovx_v = tan_fovx[v].item<float>();
	float tan_fovy_v = tan_fovy[v].item<float>();
	torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
	torch::Tensor radii = torch::zeros({P}, int_opts);  

	// Local buffers for this view
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0},  options.device(device));
	torch::Tensor imgBuffer = torch::empty({0},  options.device(device));

	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
  	if(P != 0)
  	{
		int M = 0;
		if(sh.size(0) != 0)
	    {
		  M = sh.size(1);
        }


		at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, 0);
		forwardStreams[v] = std::make_shared<at::cuda::CUDAStream>(myStream); // Store the stream for this view
		{
			at::cuda::CUDAStreamGuard guard(myStream); // Set current stream
			// launch the forward pass kernel
			rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H, // Per-view dimensions
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(), 
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix[v].contiguous().data<float>(),
			projmatrix[v].contiguous().data<float>(),
			campos[v].contiguous().data<float>(),
			tan_fovx_v,
			tan_fovy_v,
			prefiltered,
			out_color.contiguous().data<float>(),
			myStream, // Stream for this view
			radii.contiguous().data<int>(),
			debug);

            // Here we record CUDA events to signal completion of the forward pass and pass this trigger to backward pass
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, myStream.stream());
            forwardEvents[v] = event;


			// Record stream associations for tensors
            out_color.record_stream(myStream);
            radii.record_stream(myStream);
            geomBuffer.record_stream(myStream);
            binningBuffer.record_stream(myStream);
            imgBuffer.record_stream(myStream);
		

		}
		
	    // Save the results
        rendered_counts[v] = torch::tensor({rendered}, int_opts);
		out_colors[v] = out_color;
        radii_buffers[v] = radii;
        geomBuffers[v] = geomBuffer;
        binningBuffers[v] = binningBuffer;
        imgBuffers[v] = imgBuffer;
	}
  }



  return std::make_tuple(
        torch::stack(rendered_counts),
        torch::stack(out_colors),
        torch::stack(radii_buffers),
        geomBuffers,
        binningBuffers,
        imgBuffers
    );
}

std::tuple<torch::Tensor, 
torch::Tensor, 
torch::Tensor, 
torch::Tensor, 
torch::Tensor, 
torch::Tensor, 
torch::Tensor, 
torch::Tensor>
RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii, 					// Tensor with batched radii
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix, 	
    const torch::Tensor& projmatrix, 	
	const torch::Tensor& tan_fovx,   	
	const torch::Tensor& tan_fovy,   	
    const torch::Tensor& dL_dout_color, 			 // Tensor with batched gradient outputs
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos, 					 // Batched camera positions (Tensor)
	const std::vector<torch::Tensor>& geomBuffer, 	 // Batched geometry buffers (Tensor
	const torch::Tensor& R, 		  				 // Tensor with per-view rendered gaussian points counts
	const std::vector<torch::Tensor>& binningBuffer, // Batched binning buffers (Tensor)
	const std::vector<torch::Tensor>& imageBuffer,   // Batched image buffers (Tensor)
	const bool debug) 
{


  const int num_views = viewmatrix.size(0); // Number of views
  const int P = means3D.size(0); 			// Number of Gaussians
  const int H = dL_dout_color.size(2);
  const int W = dL_dout_color.size(3);

  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }


  // Allocate output tensors for gradients
  std::vector<torch::Tensor> dL_dmeans3D(num_views);
  std::vector<torch::Tensor> dL_dmeans2D(num_views);
  std::vector<torch::Tensor> dL_dcolors(num_views);
  std::vector<torch::Tensor> dL_dconic(num_views);
  std::vector<torch::Tensor> dL_dopacity(num_views);
  std::vector<torch::Tensor> dL_dcov3D(num_views);
  std::vector<torch::Tensor> dL_dsh(num_views);
  std::vector<torch::Tensor> dL_dscales(num_views);
  std::vector<torch::Tensor> dL_drotations(num_views);

  for (int v = 0; v < num_views; ++v) { 

	float tan_fovx_v = tan_fovx[v].item<float>();
    float tan_fovy_v = tan_fovy[v].item<float>(); 
  	
	// Extract per-view tensors
	auto view_projmatrix = projmatrix[v];
	auto view_viewmatrix = viewmatrix[v];
	auto view_tan_fovx = tan_fovx[v].item<float>();
	auto view_tan_fovy = tan_fovy[v].item<float>();
	auto view_geomBuffer = geomBuffer[v];
	auto view_binningBuffer = binningBuffer[v];
	auto view_imgBuffer = imageBuffer[v];
	auto view_dL_dout_color = dL_dout_color[v];
	auto view_campos = campos[v];
	auto num_rendered = R.index({v}).item<int>();


	// Allocate per-view output gradients
	dL_dmeans3D[v] = torch::zeros({P, 3}, means3D.options());
	dL_dmeans2D[v] = torch::zeros({P, 3}, means3D.options());
	dL_dcolors[v] = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	dL_dconic[v] = torch::zeros({P, 2, 2}, means3D.options());
	dL_dopacity[v] = torch::zeros({P, 1}, means3D.options());
	dL_dcov3D[v] = torch::zeros({P, 6}, means3D.options());
	dL_dsh[v] = torch::zeros({P, M, 3}, means3D.options());
	dL_dscales[v] = torch::zeros({P, 3}, means3D.options());
	dL_drotations[v] = torch::zeros({P, 4}, means3D.options());


	if(P != 0)
  	{
		auto myStreamPtr = forwardStreams[v]; // Get the stream pointer for this view

		
		{
			at::cuda::CUDAStreamGuard guard(*myStreamPtr); // Set current stream


			// Wait for the forward event for this view
			if (forwardEvents.find(v) != forwardEvents.end()) {
				cudaStreamWaitEvent(myStreamPtr->stream(), forwardEvents[v], 0);
			}
		

			// Backward computation
			CudaRasterizer::Rasterizer::backward(
				P, 
				degree, 
				M, 
				num_rendered,
			background.contiguous().data<float>(),
			W, H, 
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			view_viewmatrix.contiguous().data<float>(),
			view_projmatrix.contiguous().data<float>(),
			view_campos.contiguous().data<float>(),
			view_tan_fovx,
			view_tan_fovy,
			radii[v].contiguous().data<int>(),
			reinterpret_cast<char*>(view_geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(view_binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(view_imgBuffer.contiguous().data_ptr()),
			view_dL_dout_color.contiguous().data<float>(),
			dL_dmeans2D[v].contiguous().data<float>(),
			dL_dconic[v].contiguous().data<float>(),  
			dL_dopacity[v].contiguous().data<float>(),
			dL_dcolors[v].contiguous().data<float>(),
			dL_dmeans3D[v].contiguous().data<float>(),
			dL_dcov3D[v].contiguous().data<float>(),
			dL_dsh[v].contiguous().data<float>(),
			dL_dscales[v].contiguous().data<float>(),
			dL_drotations[v].contiguous().data<float>(),
			myStreamPtr->stream(),
			debug);	
		}
	}
  }

  for (auto& entry : forwardEvents) {
	cudaEventDestroy(entry.second);
  }
  forwardEvents.clear();

  // Synchronize all streams before stacking the results
  for (const auto& [view, streamPtr] : forwardStreams) {
	cudaStreamSynchronize(streamPtr->stream());
  }

  return std::make_tuple(torch::stack(dL_dmeans2D), 
  						 torch::stack(dL_dcolors), 
						 torch::stack(dL_dopacity), 
						 torch::stack(dL_dmeans3D), 
						 torch::stack(dL_dcov3D), 
						 torch::stack(dL_dsh), 
						 torch::stack(dL_dscales), 
						 torch::stack(dL_drotations));
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}