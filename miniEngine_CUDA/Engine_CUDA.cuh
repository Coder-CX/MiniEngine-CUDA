#pragma once

#include "Basic.h"
#include "Operators.cuh"
#include "TexDefs.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

void fillPixel1f(float* data, float color, int H, int W);
void fillPixel4f(float4* data, float4 color, int H, int W);
void convertTex4f2Tex4i(uchar4* dst, float4* src, int H, int W);
void convertTex1f2Tex4i(uchar4* dst, float* src, int H, int W);

template<class T>
__host__ __device__
inline int crossVec2(const T& v1, const T& v2)
{
	return v1(0) * v2(1) - v1(1) * v2(0);
}

template <class FS>
__global__
void fragmentProcess(FS fragmentShader, Tex4f* canvas, VertexShader* vtx_S, Box* box_S, bool* isTopLeft, int triNum,
	bool depth, Tex1f depthFrame)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	for (int triCount = 0; triCount < triNum; triCount++)
	{
		int triStart = triCount * 3;
		if (ix >= box_S[triCount].min_X && ix <= box_S[triCount].max_X && iy >= box_S[triCount].min_Y && iy <= box_S[triCount].max_Y)
		{
			VertexShader vtx[3] = { vtx_S[triStart] , vtx_S[triStart + 1], vtx_S[triStart + 2] };

			vec2i& p0 = vtx[0].pos_si;
			vec2i& p1 = vtx[1].pos_si;
			vec2i& p2 = vtx[2].pos_si;
			vec2 pos_f((float)ix + .5f, (float)iy + .5f);
			int E01 = -(ix - p0(0)) * (p1(1) - p0(1)) + (iy - p0(1)) * (p1(0) - p0(0)),
				E12 = -(ix - p1(0)) * (p2(1) - p1(1)) + (iy - p1(1)) * (p2(0) - p1(0)),
				E20 = -(ix - p2(0)) * (p0(1) - p2(1)) + (iy - p2(1)) * (p0(0) - p2(0));

			if (E01 < (isTopLeft[triStart] ? 0 : 1)) continue;
			if (E12 < (isTopLeft[triStart + 1] ? 0 : 1)) continue;
			if (E20 < (isTopLeft[triStart + 2] ? 0 : 1)) continue;

			vec2 s0 = vtx[0].pos_sf - pos_f;
			vec2 s1 = vtx[1].pos_sf - pos_f;
			vec2 s2 = vtx[2].pos_sf - pos_f;

			float a = abs(crossVec2(s1, s2));
			float b = abs(crossVec2(s2, s0));
			float c = abs(crossVec2(s0, s1));
			float s = a + b + c;
			if (s == 0.f) continue;

			a /= s;
			b /= s;
			c /= s;

			float rhw = vtx[0].rhw * a + vtx[1].rhw * b + vtx[2].rhw * c;
			if (depth && rhw < depthFrame.data[iy * canvas->W + ix])
				continue;
			depthFrame.data[iy * canvas->W + ix] = rhw;
			

			float w = 1.0f / ((rhw != 0.0f) ? rhw : 1.0f);

			float c0 = vtx[0].rhw * a * w;
			float c1 = vtx[1].rhw * b * w;
			float c2 = vtx[2].rhw * c * w;

			ContextInside fragmentInput;
			Context& input1 = vtx[0].context;
			Context& input2 = vtx[1].context;
			Context& input3 = vtx[2].context;
			
			for (int i = 0; i < MAX_CONTEXT_SIZE; i++)
			{
				
				fragmentInput.vec1f[i] = input1.vec1f[i] * c0 + input2.vec1f[i] * c1 + input3.vec1f[i] * c2;
				fragmentInput.vec2f[i] = input1.vec2f[i] * c0 + input2.vec2f[i] * c1 + input3.vec2f[i] * c2;
				fragmentInput.vec3f[i] = input1.vec3f[i] * c0 + input2.vec3f[i] * c1 + input3.vec3f[i] * c2;
				fragmentInput.vec4f[i] = input1.vec4f[i] * c0 + input2.vec4f[i] * c1 + input3.vec4f[i] * c2;
				
			}

			float4 color = fragmentShader(fragmentInput);
			canvas->data[iy * canvas->W + ix] = color;	
		}
	}
}
