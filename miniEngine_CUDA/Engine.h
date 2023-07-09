#pragma once

#include "Engine_CUDA.cuh"
#include "Basic.h"
#include "Model.h"
#include "TexDefs.h"


__host__ __device__
typedef struct Context
{
	float vec1f[MAX_CONTEXT_SIZE];
	float vec2f[MAX_CONTEXT_SIZE];
	float vec3f[MAX_CONTEXT_SIZE];
	float vec4f[MAX_CONTEXT_SIZE];
} Context;



class Engine
{
public:
	Engine(int Height, int Depth);
	virtual ~Engine() { reset(); };
	void init(int Height, int Depth);
	void reset();
	void clear1f(int frameID);
	void clear4f(int frameID);

	void fill1f(int frameID, float grayScale);
	void fill4f(int frameID, float3 RGB);
	void fill4f(int frameID, float4 RGB);

	void addFrame1f();
	void addFrame1f(int height, int width);
	void addFrame4f();
	void addFrame4f(int height, int width);

	void deleteFrame1f(int frameID);
	void deleteFrame4f(int frameID);

	template <class VS, class FS>
	void drawFrame(Model obj, VS vertexShader, FS fragmentShader);

	void saveFrame4f(int frameID, const char* file);
	

private:
	int Height, Width;
	unsigned int counter1f = 0;
	unsigned int counter4f = 0;
	bool depth = false;
	Tex1f* frame1f[MAX_FRAMEBUFFER_NUM];
	Tex4f* frame4f[MAX_FRAMEBUFFER_NUM];
	uchar4* outputFrameBuffer;
	bool usedFrame1f[MAX_FRAMEBUFFER_NUM];
	bool usedFrame4f[MAX_FRAMEBUFFER_NUM];
	Context* contextBuffer;
	Tex1f depthFrame;
};

template <class VS, class FS>
void Engine::drawFrame(Model obj, VS vertexShader, FS fragmentShader)
{

}