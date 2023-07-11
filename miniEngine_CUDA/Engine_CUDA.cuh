#pragma once

#include "Basic.h"
#include "Operators.cuh"
#include "TexDefs.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

template <typename T>
__device__
static uchar4 toUchar4(T a)
{
	return { (unsigned char)(a.x * 255), (unsigned char)(a.y * 255), (unsigned char)(a.z * 255), (unsigned char)(a.w * 255) };
}

template <>
__device__
static uchar4 toUchar4<float>(float a)
{
	return { (unsigned char)(a * 255), (unsigned char)(a * 255), (unsigned char)(a * 255), 255 };
}



template <typename T>
__global__
void convertKernel(uchar4* dst, T* src, int H, int W)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < W && iy < H)
	{
		int idx = iy * W + ix;
		dst[idx] = toUchar4<T>(src[idx]);
	}
}

void fillPixel1f(float* data, float color, int H, int W);
void fillPixel4f(float4* data, float4 color, int H, int W);
void convertTex4f2Tex4i(uchar4* dst, float4* src, int H, int W);
void convertTex1f2Tex4i(uchar4* dst, float* src, int H, int W);

template <typename T>
void convertTexFloatToInt(uchar4* dst, T* src, int H, int W)
{
	dim3 blockSize(BDIM_X, BDIM_Y);
	dim3 gridSize((W + BDIM_X - 1) / BDIM_X, (H + BDIM_Y - 1) / BDIM_Y);
	convertKernel<T> << <gridSize, blockSize >> > (dst, src, H, W);
}



