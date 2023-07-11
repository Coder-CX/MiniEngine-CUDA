#pragma once
#include "Engine_CUDA.cuh"
#include "Basic.h"
#include "Model.h"
#include "TexDefs.h"
#include "Operators.cuh"
#include <cmath>



inline float between(float val, float minV, float maxV)
{
	return std::min(std::max(val, minV), maxV);
}

inline vec4 crossVec4(const vec4& v1, const vec4& v2)
{
	vec3 _v1 = v1.head(3);
	vec3 _v2 = v2.head(3);
	vec3 tmp = _v1.cross(_v2);
	return vec4(tmp(0), tmp(1), tmp(2), v1(3));
}

inline bool isTopLeft(vec2i& v1, vec2i& v2)
{
	return ((v1(1) == v2(1)) && (v1(0) < v2(0))) || (v1(1) > v2(1));
}


class Engine
{
public:
	Engine(int Width, int Height);
	virtual ~Engine() { reset(); };

	void clear1f(int frameID);
	void clear4f(int frameID);

	void fill1f(int frameID, float grayScale);
	void fill4f(int frameID, float3 RGB);
	void fill4f(int frameID, float4 RGB);

	void addFrame1f();
	void addFrame1f(int height, int width);
	void addFrame4f();
	void addFrame4f(int height, int width);

	void useDepth(bool use)
	{
		this->depth = use;
	}

	void deleteFrame1f(int frameID);
	void deleteFrame4f(int frameID);

	template <class VS, class FS>
	bool drawFrame(Tex4f* canvas, Vertex* vtxInput, unsigned int* indices, unsigned int triNum,
		VS vertexShader, FS fragmentShader)
	{
		dim3 blockSize(BDIM_X, BDIM_Y);
		dim3 gridSize((canvas->W + BDIM_X - 1) / BDIM_X, (canvas->H + BDIM_Y - 1) / BDIM_Y);
		
		unsigned int drawCount = 0;
		
		for (unsigned int triCount = 0; triCount < triNum; triCount++)
		{
			bool vtxFault = false;
			//printf("drawCount = %d, indiceNum = %d\n", drawCount, triNum * 3);
			unsigned int vtxBufferStartIdx = drawCount * 3;
			//printf("%s: %d\n", __FILE__, __LINE__);
			for (unsigned int vtxIdx = 0; vtxIdx < 3; vtxIdx++)
			{
				unsigned int vtxBufferCount = vtxBufferStartIdx + vtxIdx;
				Vertex_S* vertex = this->vtx_S + vtxBufferCount;
				//printf("%d\n", vtxBufferCount);
				for (int i = 0; i < MAX_CONTEXT_SIZE; i++)
				{
					vertex->context.vec1f[i] = 0;
					vertex->context.vec2f[i] = { 0, 0 };
					vertex->context.vec3f[i] = { 0, 0, 0 };
					vertex->context.vec4f[i] = { 0, 0, 0, 0 };
				}
				//printf("%d: ", vtxIdx);
				//printf("%f %f %f %f\n", vertex->pos(0), vertex->pos(1), vertex->pos(2), vertex->pos(3));
				vertex->pos = vertexShader(vtxInput[indices[triCount * 3 + vtxIdx]], vertex->context);
				//printf("vtx %d: %f %f %f %f\n", vtxIdx, vertex->pos(0), vertex->pos(1), vertex->pos(2), vertex->pos(3));
				float w = vertex->pos(3);
				if (w == 0.f) { vtxFault = true; break; }
				if (vertex->pos.z() < 0.f || vertex->pos.z() > w) { vtxFault = true; break; }
				if (vertex->pos.x() < -w || vertex->pos.x() > w) { vtxFault = true; break; }
				if (vertex->pos.y() < -w || vertex->pos.y() > w) { vtxFault = true; break; }
				vertex->rhw = 1.f / w;
				vertex->pos *= vertex->rhw;
				//printf("%s: %d\n", __FILE__, __LINE__);

				vertex->pos_sf = vec2((vertex->pos(0) + 1.f) * canvas->W * 0.5f, (1.f - vertex->pos(1)) * canvas->H * 0.5f);
				vertex->pos_si = vec2i((int)(vertex->pos_sf(0) + .5f), (int)(vertex->pos_sf(1) + .5f));
				//printf("%d: %f %f\n", vtxIdx, vertex->pos_sf(0), vertex->pos_sf(1));

				Box& box = this->box[drawCount];
				if (vtxIdx == 0)
				{
					box.min_X = box.max_X = between(vertex->pos_si(0), 0, canvas->W - 1);
					box.min_Y = box.max_Y = between(vertex->pos_si(1), 0, canvas->H - 1);
				}
				else
				{
					box.min_X = between(0, canvas->W - 1, min(box.min_X, vertex->pos_si(0)));
					box.max_X = between(0, canvas->W - 1, max(box.max_X, vertex->pos_si(0)));
					box.min_Y = between(0, canvas->H - 1, min(box.min_Y, vertex->pos_si(1)));
					box.max_Y = between(0, canvas->H - 1, max(box.max_Y, vertex->pos_si(1)));
				}
			}
			if (vtxFault)
				continue;
			//printf("%s: %d, vtxBuffer[%d].pos = %f %f %f\n", __FILE__, __LINE__, drawCount, vtx_S[drawCount * 3].pos(0), vtx_S[drawCount * 3].pos(1), vtx_S[drawCount * 3].pos(2));
			vec4 v01 = vtx_S[vtxBufferStartIdx + 1].pos - vtx_S[vtxBufferStartIdx].pos;
			vec4 v02 = vtx_S[vtxBufferStartIdx + 2].pos - vtx_S[vtxBufferStartIdx].pos;
			vec4 norm = crossVec4(v01, v02);

			if (norm.z() >= 0.f)
			{
				//continue;
				VertexShader _t = vtx_S[vtxBufferStartIdx + 1];
				vtx_S[vtxBufferStartIdx + 1] = vtx_S[vtxBufferStartIdx + 2];
				vtx_S[vtxBufferStartIdx + 2] = _t;
			}
			else if (norm.z() == 0.f)
			{
				continue;
			}

			VertexShader* vtx[3] = { &vtx_S[vtxBufferStartIdx], &vtx_S[vtxBufferStartIdx + 1], &vtx_S[vtxBufferStartIdx + 2] };
				
			vec2i p0 = vtx[0]->pos_si;
			vec2i p1 = vtx[1]->pos_si;
			vec2i p2 = vtx[2]->pos_si;

			float s = std::abs(crossVec2(p1 - p0, p2 - p0));

			if (s <= 0) continue;

			this->isTopLeft_S[vtxBufferStartIdx] = isTopLeft(p0, p1);
			this->isTopLeft_S[vtxBufferStartIdx + 1] = isTopLeft(p1, p2);
			this->isTopLeft_S[vtxBufferStartIdx + 2] = isTopLeft(p2, p0);			
			
			drawCount++;
			//printf("");
			if (drawCount == MAX_TRIANGLE_BUFFER || triCount == triNum - 1)
			{
				//printf("%s %d\n", __FILE__, __LINE__);
				fragmentProcess<<<gridSize, blockSize >>> (fragmentShader, canvas, this->vtx_S, this->box, this->isTopLeft_S, drawCount, depth, depthFrame);
				cudaDeviceSynchronize();
				drawCount = 0;
			}

		}
		if (drawCount > 0)
		{
			//printf("%s %d\n", __FILE__, __LINE__);
			fragmentProcess << <gridSize, blockSize >> > (fragmentShader, canvas, this->vtx_S, this->box, this->isTopLeft_S, drawCount, depth, depthFrame);
			cudaDeviceSynchronize();
			drawCount = 0;
		}
		return true;
	}

	void saveFrame4f(int frameID, const char* file);
	void saveFrame1f(int frameID, const char* file);
	
	Tex4f* getFrame4f(int frameID);
	Tex1f* getFrame1f(int frameID);

private:
	void init(int Width, int Height);
	void reset();

	int Height, Width;
	unsigned int counter1f = 0;
	unsigned int counter4f = 0;
	bool depth = true;
	Tex1f* frame1f[MAX_FRAMEBUFFER_NUM];
	Tex4f* frame4f[MAX_FRAMEBUFFER_NUM];
	uchar4* outputFrameBuffer;
	bool usedFrame1f[MAX_FRAMEBUFFER_NUM];
	bool usedFrame4f[MAX_FRAMEBUFFER_NUM];
	Vertex_S* vtx_S;
	bool* isTopLeft_S;
	Box* box;
	Tex1f depthFrame;
};