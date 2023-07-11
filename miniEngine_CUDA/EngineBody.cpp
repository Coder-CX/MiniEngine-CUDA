
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Engine.cuh"
#include "stb_image_write.h"
#include <memory>

Engine::Engine(int width, int height)
{
	this->init(width, height);
}

void Engine::init(int width, int height)
{
	memset(usedFrame1f, 0, sizeof(bool) * MAX_FRAMEBUFFER_NUM);
	memset(usedFrame4f, 0, sizeof(bool) * MAX_FRAMEBUFFER_NUM);
	memset(frame1f, 0, sizeof(Tex1f*) * MAX_FRAMEBUFFER_NUM);
	memset(frame4f, 0, sizeof(Tex4f*) * MAX_FRAMEBUFFER_NUM);
	this->counter1f = 0;
	this->counter4f = 0;
	this->Height = height;
	this->Width = width;
	this->depthFrame.H = height;
	this->depthFrame.W = width;

	cudaMallocHost(&this->vtx_S, sizeof(Vertex_S) * MAX_TRIANGLE_BUFFER * 3);
	cudaMallocHost(&this->box, sizeof(Box) * MAX_TRIANGLE_BUFFER);
	cudaMallocHost(&this->isTopLeft_S, sizeof(bool) * MAX_TRIANGLE_BUFFER * 3);
	cudaMallocHost(&this->depthFrame.data, sizeof(float) * height * width);

	memset(this->depthFrame.data, 0, sizeof(float) * height * width);

	for (int i = 0; i < MAX_TRIANGLE_BUFFER * 3; i++)
	{
		cudaMallocHost(&this->vtx_S[i].context.vec1f, sizeof(float) * MAX_CONTEXT_SIZE);
		cudaMallocHost(&this->vtx_S[i].context.vec2f, sizeof(float2) * MAX_CONTEXT_SIZE);
		cudaMallocHost(&this->vtx_S[i].context.vec3f, sizeof(float3) * MAX_CONTEXT_SIZE);
		cudaMallocHost(&this->vtx_S[i].context.vec4f, sizeof(float4) * MAX_CONTEXT_SIZE);
	}
}

void Engine::reset()
{
	for (int i = 0; i < counter1f; i++)
	{
		if (usedFrame1f[i])
		{
			cudaFree(frame1f[i]->data);
			usedFrame1f[i] = false;
		}

	}
	this->counter1f = 0;
	for (int i = 0; i < counter4f; i++)
	{
		if (usedFrame4f[i])
		{
			cudaFree(frame4f[i]->data);
			usedFrame4f[i] = false;
		}
	}
	this->counter4f = 0;
	
	//for (int i = 0; i < MAX_TRIANGLE_BUFFER * 3; i++)
	//{
	//	free(this->vtx_S[i].context.vec1f);
	//	free(this->vtx_S[i].context.vec2f);
	//	free(this->vtx_S[i].context.vec3f);
	//	free(this->vtx_S[i].context.vec4f);
	//}
	//printf("%s, %d\n", __FILE__, __LINE__);
	//free(this->vtx_S);
	//free(this->box);
	//free(this->isTopLeft_S);
}

void Engine::clear1f(int frameID)
{
	if (usedFrame1f[frameID])
	{
		cudaMemset(frame1f[frameID]->data, 0, sizeof(float) * frame1f[frameID]->H * frame1f[frameID]->W);
	}
	else
	{
		printf("Frame_1f[%d] is empty!\n", frameID);
	}
}

void Engine::clear4f(int frameID)
{
	if (usedFrame4f[frameID])
	{
		cudaMemset(frame4f[frameID]->data, 0, sizeof(float) * frame4f[frameID]->H * frame4f[frameID]->W);
	}
	else
	{
		printf("Frame_1f[%d] is empty!\n", frameID);
	}
}

void Engine::fill1f(int frameID, float grayScale)
{
	if (usedFrame1f[frameID])
		fillPixel1f(this->frame1f[frameID]->data, grayScale, this->frame1f[frameID]->H, this->frame1f[frameID]->W);
	else
		printf("Frame_1f[%d] is empty!\n", frameID);
}

void Engine::fill4f(int frameID, float3 RGB)
{
	if (usedFrame4f[frameID])
	{
		float4 RGBD = { RGB.x, RGB.y, RGB.z, 1 };
		fillPixel4f(this->frame4f[frameID]->data, RGBD, this->frame4f[frameID]->H, this->frame4f[frameID]->W);
	}
	else
		printf("Frame_4f[%d] is empty!\n", frameID);
}

void Engine::fill4f(int frameID, float4 RGBD)
{
	if (usedFrame4f[frameID])
		fillPixel4f(this->frame4f[frameID]->data, RGBD, this->frame4f[frameID]->H, this->frame4f[frameID]->W);
	else
		printf("Frame_4f[%d] is empty!\n", frameID);
}

void Engine::addFrame1f()
{
	cudaMallocHost(frame1f + counter1f, sizeof(Tex1f));
	frame1f[counter1f]->H = this->Height;
	frame1f[counter1f]->W = this->Width;
	cudaMalloc(&(frame1f[0]->data), sizeof(float) * this->Height * this->Width);
	usedFrame1f[counter1f] = true;
	this->counter1f++;
}

void Engine::addFrame1f(int height, int width)
{
	cudaMallocHost(frame1f + counter1f, sizeof(Tex1f));
	frame1f[counter1f]->H = height;
	frame1f[counter1f]->W = width;
	cudaMalloc(&(frame1f[0]->data), sizeof(float) * height * width);
	usedFrame1f[counter1f] = true;
	this->counter1f++;
}

void Engine::addFrame4f()
{
	//frame4f[counter4f] = (Tex4f*)malloc(sizeof(Tex4f));
	cudaMallocHost(frame4f + counter4f, sizeof(Tex4f));
	frame4f[counter4f]->H = this->Height;
	frame4f[counter4f]->W = this->Width;
	cudaMalloc(&(frame4f[counter4f]->data), sizeof(float4) * this->Height * this->Width);
	usedFrame4f[counter4f] = true;
	this->counter4f++;
}

void Engine::addFrame4f(int height, int width)
{
	cudaMallocHost(frame4f + counter4f, sizeof(Tex4f));
	frame1f[counter4f]->H = height;
	frame1f[counter4f]->W = width;
	cudaMalloc(&(frame4f[0]->data), sizeof(float4) * height * width);
	usedFrame1f[counter4f] = true;
	this->counter4f++;
}

void Engine::deleteFrame1f(int frameID)
{
	if (usedFrame1f[frameID])
	{
		cudaFree(frame1f[frameID]->data);
		free(frame1f[frameID]);
		usedFrame1f[frameID] = false;
	}
	else
	{
		printf("Frame_1f[%d] is empty!\n", frameID);
	}
}

void Engine::deleteFrame4f(int frameID)
{
	if (usedFrame4f[frameID])
	{
		cudaFree(frame4f[frameID]->data);
		free(frame4f[frameID]);
		usedFrame4f[frameID] = false;
	}
	else
	{
		printf("Frame_4f[%d] is empty!\n", frameID);
	}
}

Tex1f* Engine::getFrame1f(int frameID)
{
	if (usedFrame1f[frameID])
		return this->frame1f[frameID];
	else
	{
		printf("Frame_1f[%d] is empty!\n", frameID);
		return nullptr;
	}

}

Tex4f* Engine::getFrame4f(int frameID)
{
	if (usedFrame4f[frameID])
		return this->frame4f[frameID];
	else
	{
		printf("Frame_4f[%d] is empty!\n", frameID);
		return nullptr;
	}
}

void Engine::saveFrame4f(int frameID, const char* file)
{
	if (usedFrame4f[frameID])
	{
		Tex4f* frame = this->frame4f[frameID];
		size_t dataSize = sizeof(uchar4) * frame->H * frame->W;
		uchar4* canvas_h = (uchar4*)malloc(dataSize);
		cudaMalloc(&this->outputFrameBuffer, dataSize);
		convertTex4f2Tex4i(this->outputFrameBuffer, frame->data, frame->H, frame->W);
		cudaMemcpy(canvas_h, this->outputFrameBuffer, dataSize, cudaMemcpyDeviceToHost);
		stbi_write_png(file, frame->W, frame->H, 4, canvas_h, 0);
		cudaFree(this->outputFrameBuffer);
	}
	else
	{
		printf("Frame_4f[%d] is empty!\n", frameID);
	}
}

void Engine::saveFrame1f(int frameID, const char* file)
{
	if (usedFrame1f[frameID])
	{
		Tex1f* frame = this->frame1f[frameID];
		size_t dataSize = sizeof(uchar4) * frame->H * frame->W;
		uchar4* canvas_h = (uchar4*)malloc(dataSize);
		cudaMalloc(&this->outputFrameBuffer, dataSize);
		convertTex1f2Tex4i(this->outputFrameBuffer, frame->data, frame->H, frame->W);
		cudaMemcpy(canvas_h, this->outputFrameBuffer, dataSize, cudaMemcpyDeviceToHost);
		stbi_write_png(file, frame->H, frame->W, 4, canvas_h, 0);
		cudaFree(this->outputFrameBuffer);
	}
	else
	{
		printf("Frame_4f[%d] is empty!\n", frameID);
	}
}