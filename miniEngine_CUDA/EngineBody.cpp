
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Engine.h"
#include "stb_image_write.h"
#include <memory>

Engine::Engine(int height, int width)
{
	this->init(height, width);
}

void Engine::init(int height, int width)
{
	memset(usedFrame1f, 0, sizeof(bool) * MAX_FRAMEBUFFER_NUM);
	memset(usedFrame4f, 0, sizeof(bool) * MAX_FRAMEBUFFER_NUM);
	memset(frame1f, 0, sizeof(Tex1f*) * MAX_FRAMEBUFFER_NUM);
	memset(frame4f, 0, sizeof(Tex4f*) * MAX_FRAMEBUFFER_NUM);
	this->counter1f = 0;
	this->counter4f = 0;
	this->Height = height;
	this->Width = width;
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
	frame4f[counter4f] = (Tex4f*)malloc(sizeof(Tex4f));
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
		usedFrame4f[frameID] = false;
	}
	else
	{
		printf("Frame_4f[%d] is empty!\n", frameID);
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
		stbi_write_png(file, frame->H, frame->W, 4, canvas_h, 0);
		cudaFree(this->outputFrameBuffer);
	}
	else
	{
		printf("Frame_4f[%d] is empty!\n", frameID);
	}
}