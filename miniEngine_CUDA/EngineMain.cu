#include <iostream>
#include <cuda_runtime.h>
#include "Model.h"
#include "Camera.h"
#include "EngineMath.h"
#include "Engine.cuh"
#include <cmath>

__device__
float saturatef(float& color)
{
	return __saturatef(color);
}
__device__
float saturatef(float&& color)
{
	return __saturatef(color);
}

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

__global__
void printMat(Tex1f tex, int H, int W)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < W && iy < H)
	{
		printf("%f\n", tex.data[iy * W + ix]);
	}

}

int main(void)
{
	Model obj("D:\\Code\\C\\miniEngine_CUDA\\res\\diablo3_pose_shadow.obj");
	const int H_scene = 1920, W_scene = 1080;
	const int H_s = 1024, W_s = 1024;
	Engine scene(W_scene, H_scene);
	scene.useDepth(true);
	const int outputID = 0;
	const int shadowID = 0;
	const int fakeFrameID = 1;
	const int depthID = 2;
	scene.addFrame4f(outputID);
	scene.fill4f(outputID, {0.3, 0.3, 0.3, 1});

	scene.addFrame1f(shadowID, H_s, W_s);
	scene.addFrame1f(fakeFrameID, H_s, W_s);
	scene.addFrame1f(depthID, H_scene, W_scene);
	scene.fill1f(shadowID, 0.f);
	scene.fill1f(depthID, 0.f);

	Tex1f* canvas = scene.getFrame1f(fakeFrameID);
	scene.bindDepthBuffer(shadowID);

	vec3 camPos(0 + 1, -0.5 + 1, 1.7 + 1);
	vec3 camAt(0, 0, 0);
	vec3 camUp(0, 1, 0);
	Camera cam(camPos, camAt, camUp);

	vec3 l_dir(1, 1, 0.85);
	vec3 l_pos(1, 1, 0.85);
	Camera cam_light(l_pos, camAt, camUp);
	l_dir.normalize();

	mat4 m_lookAt_l = lookAt(cam_light);
	mat4 m_proj_l = perspective(cam_light.FOV, (float)W_s / H_s, 1, 500);
	mat4 m_trans_l = m_proj_l * m_lookAt_l;

	const int COLOR = 0;

	auto VS_shadow = [=]__host__(Vertex & vtxInput, Context & context)->vec4 {
		vec4 pos = m_trans_l * vecXYZ1(vtxInput.pos);
		return pos;
	};

	auto FS_shadow = [=]__device__(ContextInside & context)->float {
		return 0.f;
	};

	for (int i = 0; i < obj.meshes.size(); i++)
	{
		scene.drawFrame<float>(canvas, obj.meshes[i].vertices_d, obj.meshes[i].indices_d, obj.meshes[i].triNum, VS_shadow, FS_shadow);
	}

	cudaTextureDesc texDesc;
	{
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;
	}

	scene.bindTex1f(0, shadowID, &texDesc);

	Tex_Shader *shadowMap_s = scene.getTexID(0);
	cudaTextureObject_t shadowMap = shadowMap_s->tex;

	const int shadow_rhw = 0;
	const int depth_rhw = 1;
	cudaTextureObject_t tex = obj.texList_d.diffuse_CUDA_Tex[0];
	cudaTextureObject_t normTex = obj.texList_d.normal_CUDA_Tex[0];

	mat4 m_lookAt = lookAt(cam);
	mat4 m_proj = perspective(cam.FOV, (float)W_scene / H_scene, 1, 500);
	mat4 m_trans = m_proj * m_lookAt;

	auto VS = [=]__host__(Vertex &vtxInput, Context &context)->vec4 {
		vec4 pos = m_trans * vecXYZ1(vtxInput.pos);
		vec4 pos_s = m_trans_l * vecXYZ1(vtxInput.pos);
		context.vec1f[shadow_rhw] = 1.f / pos_s(3);
		context.vec2f[0] = vec2ToFloat2(vtxInput.texCoord);
		context.vec2f[1] = vec2ToFloat2((pos_s / pos_s(3)).head(2));
		context.vec3f[0] = vec3ToFloat3(vtxInput.norm);
		return pos;
	};

	auto FS_1 = [=]__device__(ContextInside &context)->float4 {

		float2 uv = context.vec2f[0];
		float tex_x = uv.x * obj.texList_d.diffuse_CUDA_HW[0].y;
		float tex_y = uv.y * obj.texList_d.diffuse_CUDA_HW[0].x;
		float tex_x_s = (context.vec2f[1].x + 1.f) * W_s * 0.5f;
		float tex_y_s = (1.f - context.vec2f[1].y) * H_s * 0.5f;
		vec3 norm = float4ToVec3(tex2D<float4>(normTex, tex_x + 0.5f, tex_y + 0.5f));

		float ifShadow = 0.f;
		float bias = max(0.05 * (1.0f - norm.dot(l_dir)), 0.005);
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				ifShadow += tex2D<float>(shadowMap, tex_x_s + j, tex_y_s + i) - 0.015 >= context.vec1f[shadow_rhw] ? 0.1f : 1.f;
			}
		}
		ifShadow /= 9;
		float4 texColor = tex2D<float4>(tex, tex_x + 0.5f, tex_y + 0.5f);
		float intense = ifShadow * norm.dot(l_dir);

		return saturatef(texColor * intense);
	};

	auto FS_2 = [=]__device__(ContextInside & context)->float4 {
		float tex_x_s = (context.vec2f[1].x + 1.f) * W_s * 0.5f;
		float tex_y_s = (1.f - context.vec2f[1].y) * H_s * 0.5f;
		vec3 norm = float3ToVec3(context.vec3f[0]);
		float ifShadow = 0.f;
		float bias = max(0.05 * (1.0f - norm.dot(l_dir)), 0.005);
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				ifShadow += tex2D<float>(shadowMap, tex_x_s + j, tex_y_s + i) - bias >= context.vec1f[shadow_rhw] ? 0.1f : 1.f;
			}
		}
		ifShadow /= 9;
		
		float intense = saturatef(ifShadow * norm.dot(l_dir));

		return {intense, intense, intense, 1.f};
	};

	Tex4f* output = scene.getFrame4f(outputID);
	scene.bindDepthBuffer(depthID);


	scene.drawFrame(output, obj.meshes[0].vertices_d, obj.meshes[0].indices_d, obj.meshes[0].triNum, VS, FS_1);
	scene.drawFrame(output, obj.meshes[1].vertices_d, obj.meshes[1].indices_d, obj.meshes[1].triNum, VS, FS_2);


	string fileName = "test11.png";
	string systemCall = "mspaint.exe " + fileName;
	scene.saveFrame4f(outputID, fileName.c_str());
	system(systemCall.c_str());

	return 0;
}