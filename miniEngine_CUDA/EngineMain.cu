#include <iostream>
#include <cuda_runtime.h>
#include "Model.h"
#include "Engine.cuh"
#include "Camera.h"
#include "EngineMath.h"
#include <cmath>

int main(void)
{
	Model obj("D:\\Code\\C\\miniEngine_CUDA\\res\\diablo3_pose.obj");
	Engine scene(600, 800);
	scene.useDepth(true);
	const int outputID = 0;
	scene.addFrame4f();
	scene.fill4f(outputID, {0, 0, 0, 1});
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

	//mat4 m_model_norm = m_trans.inverse().transpose();

	const int COLOR = 0;

	auto vertexShader = [=]__host__(Vertex& vtxInput, Context& context)->vec4 {
		vec4 pos = m_trans * vecXYZ1(vtxInput.pos);
		//printf("%f %f %f %f\n", pos.x(), pos.y(), pos.z(), pos.w());
		context.vec2f[0] = vec2ToFloat2(vtxInput.texCoord);
		context.vec3f[0] = vec3ToFloat3(vtxInput.norm);
		return pos;
	};

	auto fragmentShader = [=]__device__(ContextInside& context)->float4 {
		float2 uv = context.vec2f[0];
		vec3 norm = float3ToVec3(context.vec3f[0]);
		float intense =__saturatef(norm.dot(l_dir));
		//printf("%f\n", intense);
		return { intense, intense, intense, intense };
	};
	
	for (int i = 0; i < obj.meshes.size(); i++)
	{
		scene.drawFrame(canvas, obj.meshes[i].vertices_d, obj.meshes[i].indices_d, obj.meshes[i].triNum, vertexShader, fragmentShader);
	}
	string fileName = "test6.png";
	string systemCall = "mspaint.exe " + fileName;
	scene.saveFrame4f(outputID, fileName.c_str());
	system(systemCall.c_str());

	return 0;
}