#pragma once

#include <cuda_runtime.h>
#include "Basic.h"
#include <string>

template <typename T>
struct Tex {
	int H;
	int W;
	T* data;
};

typedef struct Texture {
	textureType type;
	std::string path;
	unsigned int id;
} Texture;

using Tex1f = Tex<float>;
using Tex2f = Tex<float2>;
using Tex3f = Tex<float3>;
using Tex4f = Tex<float4>;

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
	unsigned int numDiffuse;
} TexInput;