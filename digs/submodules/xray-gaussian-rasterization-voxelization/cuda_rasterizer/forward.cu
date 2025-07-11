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
#include <stdio.h> // for debug
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward version of 2D covariance matrix computation
// Section 6.2 EWA Volume Resampling Filter
__device__ float4 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, const int mode)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	//! This is the transformation from Source (world) to view (camera)
	float3 t = transformPoint4x3(mean, viewmatrix);
    glm::mat3 J;
	if (mode == 0){  // parallel beam
		const float limx = 1.3f;
		const float limy = 1.3f;
		t.x = min(limx, max(-limx, t.x));
		t.y = min(limx, max(-limx, t.y));

		// J is eye
		J = glm::mat3(
		focal_x, 0.0f, 0.0f,
		0.0f, focal_y, 0.0f,
		0.0f, 0.0f, 1.0f);
	}
	else  // cone beam
	{	
		const float limx = 1.3f * tan_fovx;
		const float limy = 1.3f * tan_fovy;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		// t = Phi^-1(x), eq(26)
		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;
		// Jacobian of Affine approximation of projection transformation
		// eq(29)
		const float l = sqrt(t.x * t.x +  t.y * t.y + t.z * t.z);
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			t.x / l, t.y / l, t.z / l);  //! We add the third row for further computation.
	}
	
	// Viewing transformation
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	
	glm::mat3 M = W * J;
    
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
    
	// J W Sigma W^M J^M
	glm::mat3 cov = glm::transpose(M) * glm::transpose(Vrk) * M;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	//! We do not add low pass filter.
	// cov[0][0] += 0.0f;
	// cov[1][1] += 0.0f;

	//! Compute integration bias factor mu (Eq. 7 in our paper) 
	//! Check issue #4 regarding dicussion of ray-space and world-space scales.
	// float hata = cov[0][0];
	// float hatb = cov[0][1];
	// float hatc = cov[0][2];
	// float hatd = cov[1][1];
	// float hate = cov[1][2];
	// float hatf = cov[2][2];
	// float diamond = hata * hatd - hatb * hatb;
	// float circ = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	// float mu_square = 2 * M_PI * circ / diamond;
	float mu = 0.0f;
	float circ_diamond = cov[2][2] - (cov[1][1]*cov[0][2]*cov[0][2]-2.0f*cov[0][1]*cov[0][2]*cov[1][2]+cov[0][0]*cov[1][2]*cov[1][2])/(cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
	if (circ_diamond > 0.0f){
		mu = sqrt(2 * M_PI * circ_diamond);  //TODO Sometimes mu_square becomes negative, may need to check why.
	}

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]), float(mu) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];  // a
	cov3D[1] = Sigma[0][1];  // b
	cov3D[2] = Sigma[0][2];  // c
	cov3D[3] = Sigma[1][1];  // d
	cov3D[4] = Sigma[1][2];  // e
	cov3D[5] = Sigma[2][2];  // f
}

//deform version of computeCov3D
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, const float* deformJacobian, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 J = glm::mat3(
		deformJacobian[0], deformJacobian[1], deformJacobian[2],
		deformJacobian[3], deformJacobian[4], deformJacobian[5],
		deformJacobian[6], deformJacobian[7], deformJacobian[8]
	);

	glm::mat3 M = S * R * J;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];  // a
	cov3D[1] = Sigma[0][1];  // b
	cov3D[2] = Sigma[0][2];  // c
	cov3D[3] = Sigma[1][1];  // d
	cov3D[4] = Sigma[1][2];  // e
	cov3D[5] = Sigma[2][2];  // f
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* deformJacobians,
	const float* opacities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* cov3Ds,
	float4* conic_opacity,
	float* mus,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const int mode
	)
{
	auto idx = cg::this_grid().thread_rank(); //idx
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		if (deformJacobians == nullptr)
		{
			computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		}
		else
		{
			computeCov3D(scales[idx], scale_modifier, rotations[idx], deformJacobians + idx * 9, cov3Ds + idx * 6);
		}
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float4 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, mode);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); 
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// if (my_radius <= 0)
	// 	return;
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store some useful helper data for the next steps.
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	mus[idx] = cov.w;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ mus,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_mu[BLOCK_SIZE];

	// Initialize helper variables
	float accum = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_mu[block.thread_rank()] = mus[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; inside && j < min(BLOCK_SIZE, toDo); j++)
		{

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float mu = collected_mu[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			//this makes the rasterization faster
			const float alpha = con_o.w * mu * exp(power);
			if (alpha < 0.00001f)
				continue;

			//! We compute mu to consider integration.
			accum += alpha;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		out_color[pix_id] = accum;
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float* mus,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		mus,
		out_color);
}


void FORWARD::preprocess(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* deformJacobians,
	const float* opacities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* cov3Ds,
	float4* conic_opacity,
	float* mus,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const int mode
	)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		deformJacobians,
		opacities,
		cov3D_precomp,
		viewmatrix, 
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		cov3Ds,
		conic_opacity,
		mus,
		grid,
		tiles_touched,
		prefiltered,
		mode
		);
}