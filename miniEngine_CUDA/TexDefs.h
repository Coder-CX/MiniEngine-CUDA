#pragma once

#include <cuda_runtime.h>


typedef struct Texture {
	textureType type;
	string path;
	unsigned int id;
} Texture;

typedef struct _Tex1f {
	int H;
	int W;
	float* data;
} Tex1f;

typedef struct _Tex2f {
	int H;
	int W;
	float2* data;
} Tex2f;

typedef struct _Tex3f {
	int H;
	int W;
	float3* data;
} Tex3f;

typedef struct _Tex4f {
	int H;
	int W;
	float4* data;
} Tex4f;

typedef struct _TextureDataList {
	vector<Tex4f> DiffuseMap;
	vector<Tex4f> NormalMap;
	vector<Tex1f> OpacityMap;
} TextureDataList;

typedef struct _TextureUsage {
	bool DiffuseTex = false;
	bool ShinessTex = false;
	bool NormalTex = false;
	bool OpacityTex = false;
} TextureUsage;

typedef struct _TexID {
	vector<unsigned int> diffuseMap;
	unsigned int normalMap;
	unsigned int opacityMap;
} TexID;

typedef struct _Tex_s {
	int tex_num;
	int* idDiffuseMap;
	int2* tex_HW;
	int2 normal_HW;
	cudaTextureObject_t normal;
	cudaTextureObject_t* tex;
} Tex_s;

typedef struct _TexList_CUDA {
	int2* diffuse_CUDA_HW;
	int2* normal_CUDA_HW;
	int2* opacity_CUDA_HW;
	cudaTextureObject_t* normal_CUDA_Tex;
	cudaTextureObject_t* opacity_CUDA_Tex;
	cudaTextureObject_t* diffuse_CUDA_Tex;
} TexList;

typedef struct _TexInput {
	bool useDiffuse, useNormal, useOpacity;
	unsigned int ID_normal, ID_opacity, *ID_diffuse;
	int numDiffuse;
} TexInput;