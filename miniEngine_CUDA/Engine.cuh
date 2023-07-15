#pragma once

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Engine_CUDA.cuh"
#include "fragmentShader.cuh"
#include "Basic.h"
#include "Model.h"
#include "TexDefs.h"
#include "Operators.cuh"
#include "stb_image_write.h"
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

	void addFrame1f(int frameID, int height = -1, int width = -1)
	{
		addFrame<float>(frameID, this->frame1f, this->usedFrame1f, height, width);
	}
	void addFrame4f(int frameID, int height = -1, int width = -1)
	{
		addFrame<float4>(frameID, this->frame4f, this->usedFrame4f, height, width);
	}

	void useDepth(bool use)
	{
		this->depth = use;
	}

	void useBackCulling(bool use)
	{
		this->backCulling = use;
	}
	
	void deleteFrame1f(int frameID)
	{
		deleteFrame<float>(frameID, frame1f, usedFrame1f);
	}
	void deleteFrame4f(int frameID)
	{
		deleteFrame<float4>(frameID, frame4f, usedFrame4f);
	}

	template <typename T, class VS, class FS>
	bool drawFrame(Tex<T>* canvas, Vertex* vtxInput, unsigned int* indices, unsigned int triNum,
		VS vertexShader, FS fragmentShader)
	{
		dim3 blockSize(BDIM_X, BDIM_Y);
		dim3 gridSize((canvas->W + BDIM_X - 1) / BDIM_X, (canvas->H + BDIM_Y - 1) / BDIM_Y);
		
		unsigned int drawCount = 0;
		
		for (unsigned int triCount = 0; triCount < triNum; triCount++)
		{
			bool vtxFault = false;
			unsigned int vtxBufferStartIdx = drawCount * 3;
			for (unsigned int vtxIdx = 0; vtxIdx < 3; vtxIdx++)
			{
				unsigned int vtxBufferCount = vtxBufferStartIdx + vtxIdx;
				Vertex_S* vertex = this->vtx_S + vtxBufferCount;
				for (int i = 0; i < MAX_CONTEXT_SIZE; i++)
				{
					vertex->context.vec1f[i] = 0;
					vertex->context.vec2f[i] = { 0, 0 };
					vertex->context.vec3f[i] = { 0, 0, 0 };
					vertex->context.vec4f[i] = { 0, 0, 0, 0 };
				}
				vertex->pos = vertexShader(vtxInput[indices[triCount * 3 + vtxIdx]], vertex->context);
				float w = vertex->pos(3);
				if (w == 0.f) { vtxFault = true; break; }
				if (vertex->pos.z() < 0.f || vertex->pos.z() > w) { vtxFault = true; break; }
				if (vertex->pos.x() < -w || vertex->pos.x() > w) { vtxFault = true; break; }
				if (vertex->pos.y() < -w || vertex->pos.y() > w) { vtxFault = true; break; }
				vertex->rhw = 1.f / w;
				vertex->pos *= vertex->rhw;
				
				vertex->pos_sf = vec2((vertex->pos(0) + 1.f) * canvas->W * 0.5f, (1.f - vertex->pos(1)) * canvas->H * 0.5f);
				vertex->pos_si = vec2i((int)(vertex->pos_sf(0) + .5f), (int)(vertex->pos_sf(1) + .5f));
				
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
			vec4 v01 = vtx_S[vtxBufferStartIdx + 1].pos - vtx_S[vtxBufferStartIdx].pos;
			vec4 v02 = vtx_S[vtxBufferStartIdx + 2].pos - vtx_S[vtxBufferStartIdx].pos;
			vec4 norm = crossVec4(v01, v02);

			if (this->backCulling && norm.z() >= 0.f)
				continue;
			else
			{
				if (norm.z() > 0.f)
				{
					VertexShader _t = vtx_S[vtxBufferStartIdx + 1];
					vtx_S[vtxBufferStartIdx + 1] = vtx_S[vtxBufferStartIdx + 2];
					vtx_S[vtxBufferStartIdx + 2] = _t;
				}
				else if (norm.z() == 0.f)
				{
					continue;
				}
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
			if (drawCount == MAX_TRIANGLE_BUFFER || triCount == triNum - 1)
			{
				fragmentProcess <<<gridSize, blockSize >>> (canvas, fragmentShader, this->vtx_S, this->box, this->isTopLeft_S, drawCount, depth, depthFrame);
				cudaDeviceSynchronize();
				drawCount = 0;
			}

		}
		if (drawCount > 0)
		{
			fragmentProcess << <gridSize, blockSize >> > (canvas, fragmentShader, this->vtx_S, this->box, this->isTopLeft_S, drawCount, depth, depthFrame);
			cudaDeviceSynchronize();
			drawCount = 0;
		}
		return true;
	}

	void saveFrame1f(int frameID, const char* file)
	{
		saveFrame<float>(frameID, file, frame1f, usedFrame1f);
	}
	void saveFrame4f(int frameID, const char* file)
	{
		saveFrame<float4>(frameID, file, frame4f, usedFrame4f);
	}
	
	Tex4f* getFrame4f(int frameID);
	Tex1f* getFrame1f(int frameID);
	Tex_Shader* getTexID(int texID)
	{
		if (usedTexID[texID])
			return this->texList + texID;
		else
			printf("Tex [%d] is Used.\n", texID);
	}
	Tex1f getDepthFrame(void)
	{
		return this->depthFrame;
	}

	void bindTex1f(int texID, int frameID, cudaTextureDesc* texDesc)
	{
		bindTex<float>(texID, frameID, texDesc, this->frame1f, this->usedFrame1f, &this->channelDesc_1f);
	}
	void bindTex4f(int texID, int frameID, cudaTextureDesc* texDesc)
	{
		bindTex<float4>(texID, frameID, texDesc, this->frame4f, this->usedFrame4f, &this->channelDesc_4f);
	}

	void deleteTex(int texID);
	void bindDepthBuffer(int frameID)
	{
		if (usedFrame1f[frameID])
		{
			this->depthFrame.H = this->frame1f[frameID]->H;
			this->depthFrame.W = this->frame1f[frameID]->W;
			this->depthFrame.data = this->frame1f[frameID]->data;
		}
		else
		{
			printf("Frame [%d] is Empty.\n", frameID);
		}
	}


private:
	void init(int Width, int Height);
	void reset();

	template <typename T>
	void addFrame(int frameID, Tex<T>** frame, bool* usedFrame, int height = -1, int width = -1)
	{
		if (!usedFrame[frameID])
		{
			cudaMallocHost(frame + frameID, sizeof(Tex<T>));
			frame[frameID]->H = height == -1 ? this->Height : height;
			frame[frameID]->W = width == -1 ? this->Width : width;
			cudaMalloc(&(frame[frameID]->data), sizeof(T) * frame[frameID]->H * frame[frameID]->W);
			usedFrame[frameID] = true;
		}
		else
		{
			printf("Frame [%d] is Used.\n", frameID);
		}
	}
	
	template <typename T>
	void deleteFrame(int frameID, Tex<T>** frame, bool* usedFrame)
	{
		if (usedFrame[frameID])
		{
			cudaFree(frame[frameID]->data);
			//free(frame + frameID);
			usedFrame[frameID] = false;
		}
		else
		{
			printf("Frame [%d] is empty!\n", frameID);
		}
	}

	template <typename T>
	void saveFrame(int frameID, const char* file, Tex<T>** frame, bool* usedFrame)
	{
		if (usedFrame[frameID])
		{
			Tex<T>* _frame = frame[frameID];
			size_t dataSize = sizeof(uchar4) * _frame->H * _frame->W;
			uchar4* canvas_h = (uchar4*)malloc(dataSize);
			cudaMalloc(&this->outputFrameBuffer, dataSize);

			convertTexFloatToInt<T>(this->outputFrameBuffer, _frame->data, _frame->H, _frame->W);
			cudaMemcpy(canvas_h, this->outputFrameBuffer, dataSize, cudaMemcpyDeviceToHost);
			stbi_write_png(file, _frame->W, _frame->H, 4, canvas_h, 0);
			cudaFree(this->outputFrameBuffer);
		}
		else
		{
			printf("Frame [%d] is empty!\n", frameID);
		}
	}

	template <typename T>
	void bindTex(int texID, int frameID, cudaTextureDesc* texDesc, Tex<T>** frame, bool* usedFrame, cudaChannelFormatDesc* channelDesc)
	{
		if (usedFrame[frameID] && !usedTexID[texID])
		{
			const int H = frame[frameID]->H;
			const int W = frame[frameID]->W;
			CHECK(cudaMallocArray(&this->texList[texID].tex_data, channelDesc, W, H));
			cudaMemcpy2DToArray(this->texList[texID].tex_data, 0, 0, frame[frameID]->data, sizeof(T) * W, sizeof(T) * W, H, cudaMemcpyDeviceToDevice);

			struct cudaResourceDesc resDesc;
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = this->texList[texID].tex_data;
			cudaCreateTextureObject(&this->texList[texID].tex, &resDesc, texDesc, NULL);
			this->usedTexID[texID] = true;
		}
		else if (!usedFrame[frameID])
		{
			printf("Frame [%d] is Empty!\n", frameID);
		}
		else
		{
			printf("tex [%d] is Used!\n", texID);
		}
	}

private:
	int Height, Width;

	unsigned int counter1f = 0;
	unsigned int counter4f = 0;

	bool depth = true;
	bool backCulling = true;
	Tex1f depthFrame;
	bool usedFrame1f[MAX_FRAMEBUFFER_NUM];
	bool usedFrame4f[MAX_FRAMEBUFFER_NUM];
	Tex1f* frame1f[MAX_FRAMEBUFFER_NUM];
	Tex4f* frame4f[MAX_FRAMEBUFFER_NUM];
	uchar4* outputFrameBuffer;

	Vertex_S* vtx_S;
	bool* isTopLeft_S;
	Box* box;

	bool usedTexID[2 * MAX_FRAMEBUFFER_NUM];
	Tex_Shader texList[2 * MAX_FRAMEBUFFER_NUM];

	cudaChannelFormatDesc channelDesc_1f = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_2f = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_3f = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_4f = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

};