#include <iostream>
#include <cuda_runtime.h>
#include "Model.h"
#include "Engine.cuh"

int main(void)
{
	Vertex* vtxInput;
	cudaMallocHost(&vtxInput, sizeof(Vertex) * 3);
	vtxInput[0].pos = vec3(0.0, 0.7, 0.90);
	vtxInput[1].pos = vec3(-0.6, -0.2, 0.01);
	vtxInput[2].pos = vec3(+0.6, -0.2, 0.01);

	unsigned int* indices;
	cudaMallocHost(&indices, sizeof(unsigned int) * 3);
	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;

	Engine scene(1920, 1080);

	const int outputID = 0;
	scene.addFrame4f();
	scene.fill4f(outputID, {0, 0, 0, 1});

	const int COLOR = 0;

	auto vertexShader = [=]__host__(Vertex& vtxInput, Context& context)->vec4 {
		context.vec4f[COLOR] = {1.f, 1.f, 1.f, 1.f};
		vec3& pos = vtxInput.pos;
		return vec4(pos(0), pos(1), pos(2), 1.f);
	};

	auto fragmentShader = [=]__device__(ContextInside& context)->float4 {
		return context.vec4f[COLOR];
	};
	Tex4f* canvas = scene.getFrame4f(outputID);
	scene.drawFrame(canvas, vtxInput, indices, 1, vertexShader, fragmentShader);
	scene.saveFrame4f(outputID, "test.png");
	system("mspaint.exe test.png");

	return 0;
}