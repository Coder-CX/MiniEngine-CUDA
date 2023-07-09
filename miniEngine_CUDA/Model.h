#pragma once
#include "Basic.h"
#include "stb_image.h"
#include "TexDefs.h"
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

typedef struct _TexIDCount {
	unsigned int idDiffuseMap = 0;
	unsigned int idNormalMap = 0;
	unsigned int idOpacityMap = 0;
} TexIDCount;

struct Vertex
{
	vec3 pos;
	vec2 texCoord;
	vec3 norm;
	vec3 tangent;
	vec3 biTangent;
};

class Mesh {
public:
	vector<Vertex> vertices;
	vector<unsigned int> indices;
	unsigned int faceNum;
	
	Vertex* vertices_d;
	unsigned int* indices_d;

	float opacity = 1.f;
	TexID texID;
	TexInput texID_d;

	Mesh(vector<Vertex> vertices, vector<unsigned int> indices, TexID texID, float Opacity = 1.f)
	{
		this->vertices = vertices;
		this->indices = indices;
		this->texID = texID;
		this->opacity = Opacity;
		cudaInit();
	}
private:
	void cudaInit()
	{
		cudaMalloc(&vertices_d, sizeof(Vertex) * faceNum);		
		cudaMalloc(&indices_d, sizeof(unsigned int) * faceNum * 3);

		cudaMemset(vertices_d, 0, sizeof(Vertex) * faceNum);
		cudaMemset(indices_d, 0, sizeof(unsigned int) * faceNum * 3);

		cudaMemcpy(vertices_d, vertices.data(), sizeof(Vertex) * faceNum, cudaMemcpyHostToDevice);
		cudaMemcpy(indices_d, indices.data(), sizeof(unsigned int) * faceNum * 3, cudaMemcpyHostToDevice);

		if (texID.diffuseMap.size() > 0)
		{
			texID_d.useDiffuse = true;
			texID_d.numDiffuse = texID.diffuseMap.size();
			cudaMallocHost(&texID_d.ID_diffuse, sizeof(unsigned int) * texID_d.numDiffuse);
			memcpy(texID_d.ID_diffuse, texID.diffuseMap.data(), sizeof(unsigned int) * texID_d.numDiffuse);
		}
		if (texID.normalMap)
		{
			texID_d.ID_normal = texID.normalMap;
			texID_d.useNormal = true;
		}

	}
};

static Eigen::MatrixXi TextureFromFile(const char* path, const string& directory)
{
	string filename = string(path);
	filename = directory + '/' + filename;
	int width, height, nrComponents;
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
	Eigen::MatrixXi img(height, width);
	img = Eigen::MatrixXi::Zero(height, width);
	if (data)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				img(i, j) = (unsigned short)data[(i * width) + j];
			}
		}
		stbi_image_free(data);
	}
	else
	{
		std::cout << "Texture failed to load at path: " << path << std::endl;
		stbi_image_free(data);
	}

	return img;
}

class Model {
public:
	vector<Mesh> meshes;
	vector<Texture> loadedDiffuse;
	vector<Texture> loadedNormal;
	vector<Texture> loadedOpacity;
	bool flagLoadedDiffuse[64];
	bool flagLoadedNormal[64];
	bool flagLoadedOpacity[64];
	TexList texList_d;
	TextureDataList texList;

	string directory;

	Model(string const& path)
	{
		memset(flagLoadedDiffuse, 0, 64 * sizeof(bool));
		memset(flagLoadedNormal, 0, 64 * sizeof(bool));
		memset(flagLoadedOpacity, 0, 64 * sizeof(bool));
		cudaInit();
		Import(path);
		//cout << texList.DiffuseMap[0].H << " " << texList.DiffuseMap[0].W << endl;
		//showIntensity(texList.DiffuseMap[1].data, texList.DiffuseMap[1].H, texList.DiffuseMap[1].W, false);
	}

private:

	void cudaInit()
	{
		memset(&texDesc, 0, sizeof(struct cudaTextureDesc));
		memset(&texDesc_tex, 0, sizeof(struct cudaTextureDesc));
		//memset(&resDesc_depth, 0, sizeof(resDesc_depth));
		this->texDesc.addressMode[0] = cudaAddressModeBorder;
		this->texDesc.addressMode[1] = cudaAddressModeBorder;
		this->texDesc.filterMode = cudaFilterModeLinear;
		this->texDesc.readMode = cudaReadModeElementType;
		this->texDesc.normalizedCoords = 0;

		this->texDesc_tex.addressMode[0] = cudaAddressModeBorder;
		this->texDesc_tex.addressMode[1] = cudaAddressModeBorder;
		this->texDesc_tex.filterMode = cudaFilterModeLinear;
		this->texDesc_tex.readMode = cudaReadModeElementType;
		this->texDesc_tex.normalizedCoords = 0;
	}

	// Texture Describe
	struct cudaTextureDesc  texDesc;
	struct cudaTextureDesc  texDesc_tex;

	// normalMap buffers
	cudaArray_t* normal_CUDA;
	struct cudaResourceDesc* resDesc_normal;
	// opacityMap buffers
	cudaArray_t* opacity_CUDA;
	struct cudaResourceDesc* resDesc_opacity;
	// diffuseMap buffers
	cudaArray_t* diffuse_CUDA;
	struct cudaResourceDesc* resDesc_diffuse;

	cudaChannelFormatDesc channelDesc_1f = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_2f = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_3f = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_4f = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	bool Import(const string& file);

	void processNode(aiNode* node, const aiScene* scene, TexIDCount& texID);

	Mesh processMesh(aiMesh* mesh, const aiScene* scene, TexIDCount& texID);

	vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, textureType typeName, TexIDCount& texID);

	TexID loadTexture(vector<Texture>& textures);

};

