#include <iostream>
#include <cuda_runtime.h>
#include "Model.h"
#include "Camera.h"
#include "EngineMath.h"
#include "Engine.cuh"
#include <cmath>

__device__
float4 saturatef(float4& color)
{
	return { __saturatef(color.x),__saturatef(color.y), __saturatef(color.z), __saturatef(color.w) };
}
__device__
float4 saturatef(float4&& color)
{
	return { __saturatef(color.x),__saturatef(color.y), __saturatef(color.z), __saturatef(color.w) };
}


int main(void)
{
	Model obj("D:\\Code\\C\\miniEngine_CUDA\\res\\diablo3_pose.obj");
	Engine scene(1080, 1920);
	scene.useDepth(true);
	const int outputID = 0;
	scene.addFrame4f(outputID);
	scene.fill4f(outputID, {0.3, 0.3, 0.3, 1});
	Tex4f* canvas = scene.getFrame4f(outputID);
	vec3 camPos(0, -0.5, 1.7);
	vec3 camAt(0, 0, 0);
	vec3 camUp(0, 1, 0);
	Camera cam(camPos, camAt, camUp);
	vec3 l_dir(1, 1, 0.85);
	l_dir.normalize();
	mat4 m_lookAt = lookAt(cam);
	mat4 m_proj = perspective(cam.FOV, (float)canvas->W / canvas->H, 1, 500);
	mat4 m_trans = m_proj * m_lookAt;
	cudaTextureObject_t tex = obj.texList_d.diffuse_CUDA_Tex[0];
	
	const int COLOR = 0;

	auto vertexShader = [=]__host__(Vertex& vtxInput, Context& context)->vec4 {
		vec4 pos = m_trans * vecXYZ1(vtxInput.pos);
		context.vec2f[0] = vec2ToFloat2(vtxInput.texCoord);
		context.vec3f[0] = vec3ToFloat3(vtxInput.norm);
		return pos;
	};

	auto fragmentShader = [=]__device__(ContextInside& context)->float4 {
		float2 uv = context.vec2f[0];
		float tex_x = uv.x * obj.texList_d.diffuse_CUDA_HW[0].y;
		float tex_y = uv.y * obj.texList_d.diffuse_CUDA_HW[0].x;
		vec3 norm = float3ToVec3(context.vec3f[0]);
		float intense = norm.dot(l_dir);
		float4 texColor = tex2D<float4>(tex, tex_x + 0.5f, tex_y + 0.5f);
		return saturatef(texColor * intense);
	};
	
	clock_t tic, toc;
	tic = clock();
	for (int t = 0; t < 60; t++)
	{
		for (int i = 0; i < obj.meshes.size(); i++)
		{
			scene.drawFrame(canvas, obj.meshes[i].vertices_d, obj.meshes[i].indices_d, obj.meshes[i].triNum, vertexShader, fragmentShader);
		}
	}
	toc = clock();
	printf("time = %f s\n", (toc - tic) / 1000.f);


	string fileName = "test2.png";
	string systemCall = "mspaint.exe " + fileName;
	scene.saveFrame4f(outputID, fileName.c_str());
	system(systemCall.c_str());

	return 0;
}