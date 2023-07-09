#include "Engine_CUDA.cuh"

template <typename T>
__global__
void fillPixelKernel(T* data, T color, int H, int W)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < W && iy < H)
	{
		data[iy * W + ix] = color;
	}
}

__device__ 
uchar4 float4_uchar4(float4 a)
{
	return {(unsigned char)(a.x * 255), (unsigned char)(a.y * 255), (unsigned char)(a.z * 255), (unsigned char)(a.w * 255) };
}

__global__
void convertKernel(uchar4* dst, float4* src, int H, int W)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < W && iy < H)
	{
		int idx = iy * W + ix;
		dst[idx] = float4_uchar4(src[idx]);
	}
}

void fillPixel1f(float* data, float color, int H, int W)
{
	dim3 blockSize(BDIM_X, BDIM_Y);
	dim3 gridSize((W + BDIM_X - 1) / BDIM_X, (H + BDIM_Y - 1) / BDIM_Y);
	fillPixelKernel<< <gridSize, blockSize >> > (data, color, H, W);
}

void fillPixel4f(float4* data, float4 color, int H, int W)
{
	dim3 blockSize(BDIM_X, BDIM_Y);
	dim3 gridSize((W + BDIM_X - 1) / BDIM_X, (H + BDIM_Y - 1) / BDIM_Y);
	fillPixelKernel<< <gridSize, blockSize >> > (data, color, H, W);
}

void convertTex4f2Tex4i(uchar4* dst, float4* src, int H, int W)
{
	dim3 blockSize(BDIM_X, BDIM_Y);
	dim3 gridSize((W + BDIM_X - 1) / BDIM_X, (H + BDIM_Y - 1) / BDIM_Y);
	convertKernel << <gridSize, blockSize >> > (dst, src, H, W);
}