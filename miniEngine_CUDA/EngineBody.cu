#include "Engine.cuh"

#include <memory>

Engine::Engine(int width, int height)
{
	this->init(width, height);
}

void Engine::init(int width, int height)
{
	memset(usedFrame1f, 0, sizeof(bool) * MAX_FRAMEBUFFER_NUM);
	memset(usedFrame4f, 0, sizeof(bool) * MAX_FRAMEBUFFER_NUM);
	memset(usedTexID, 0, sizeof(float) * MAX_FRAMEBUFFER_NUM * 2);

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
	for (int i = 0; i < MAX_FRAMEBUFFER_NUM; i++)
	{
		//printf("%s %d\n", __FILE__, __LINE__);
		if (usedFrame1f[i])
			deleteFrame1f(i);
		if (usedFrame4f[i])
			deleteFrame4f(i);
	}
	this->counter1f = 0;
	this->counter4f = 0;

	for (int i = 0; i < 2 * MAX_FRAMEBUFFER_NUM; i++)
	{
		if (usedTexID[i])
			deleteTex(i);
	}
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

void Engine::deleteTex(int texID)
{
	if (this->usedTexID[texID])
	{
		cudaFreeArray(this->texList[texID].tex_data);
		cudaDestroyTextureObject(this->texList[texID].tex);
		this->usedTexID[texID] = false;
	}
	else
	{
		printf("Tex [%d] is Empty!\n", texID);
	}
}