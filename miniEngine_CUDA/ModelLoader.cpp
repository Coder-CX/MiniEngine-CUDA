#include "Basic.h"
#include "Model.h"
#include "PixelDef.h"
#include "TexDefs.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool Model::Import(const string& file)
{
	Assimp::Importer importer;
	unsigned int t = (aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
	const aiScene* scene = importer.ReadFile(file, t);
	TexIDCount texID;

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		return false;
	}

	directory = file.substr(0, file.find_last_of('/'));
	processNode(scene->mRootNode, scene, texID);
	return true;
}

void Model::processNode(aiNode* node, const aiScene* scene, TexIDCount& texID)
{
	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene, texID));
	}
	for (unsigned int i = 0; i < node->mNumChildren; i++)
		processNode(node->mChildren[i], scene, texID);
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene, TexIDCount& texID)
{
	vector<Vertex> vertices;
	vector<unsigned int> indices;
	vector<Texture> textures;
	vector<unsigned int> N_gon;

	for (unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		Vertex vertex;
		vec3 vector;
		vector(0) = mesh->mVertices[i].x;
		vector(1) = mesh->mVertices[i].z; // Switch Y-Z Axises
		vector(2) = mesh->mVertices[i].y; // Switch Y-Z Axises
		vertex.pos = vector;
		//Normals
		if (mesh->HasNormals())
		{
			vector(0) = mesh->mNormals[i].x;
			vector(1) = mesh->mNormals[i].z; // Switch Y-Z Axises
			vector(2) = mesh->mNormals[i].y; // Switch Y-Z Axises
			vertex.norm = vector;
		}
		//Texture coordnates
		if (mesh->mTextureCoords[0])
		{
			vec2 vec;
			vec(0) = mesh->mTextureCoords[0][i].x;
			vec(1) = mesh->mTextureCoords[0][i].y;
			vertex.texCoord = vec;

			vector(0) = mesh->mTangents[i].x;
			vector(1) = mesh->mTangents[i].z; // Switch Y-Z Axises
			vector(2) = mesh->mTangents[i].y; // Switch Y-Z Axises
			vertex.tangent = vector;

			vector(0) = mesh->mBitangents[i].x;
			vector(1) = mesh->mBitangents[i].z; // Switch Y-Z Axises
			vector(2) = mesh->mBitangents[i].y; // Switch Y-Z Axises
			vertex.biTangent = vector;
		}
		else
			vertex.texCoord = vec2::Zero();

		vertices.push_back(vertex);
	}

	for (unsigned int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		N_gon.push_back(face.mNumIndices);
		for (unsigned int j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}

	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

	float opacity;
	material->Get(AI_MATKEY_OPACITY, opacity);

	// 1. diffuse maps
	vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, DIFFUSE_MAP, texID);
	textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
	// 2. specular maps
	vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, SPECULAR_MAP, texID);
	textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
	// 3. normal maps
	std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, NORMAL_MAP, texID);
	textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
	// 4. height maps
	std::vector<Texture> heightMaps = loadMaterialTextures(material, aiTextureType_AMBIENT, HEIGHT_MAP, texID);
	textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

	std::vector<Texture> opacityMaps = loadMaterialTextures(material, aiTextureType_OPACITY, OPACITY_MAP, texID);
	textures.insert(textures.end(), opacityMaps.begin(), opacityMaps.end());

	TexID texIDMesh = loadTexture(textures);

	return Mesh(vertices, indices, texIDMesh, opacity);
}

vector<Texture> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, textureType typeName, TexIDCount& texID)
{
	vector<Texture> textures;
	for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
	{
		Texture texture;
		aiString str;
		mat->GetTexture(type, i, &str);
		texture.type = typeName;
		texture.path = str.C_Str();


		bool skip = false;
		switch (typeName)
		{
		case DIFFUSE_MAP:
			for (unsigned int j = 0; j < loadedDiffuse.size(); j++)
			{
				if (strcmp(loadedDiffuse[j].path.data(), str.C_Str()) == 0)
				{
					texture.id = j;
					textures.push_back(texture);
					skip = true;
					break;
				}
			}
			break;
		case NORMAL_MAP:
			for (unsigned int j = 0; j < loadedNormal.size(); j++)
			{
				if (strcmp(loadedNormal[j].path.data(), str.C_Str()) == 0)
				{
					texture.id = j;
					textures.push_back(texture);
					skip = true;
					break;
				}
			}
			break;
		case OPACITY_MAP:
			for (unsigned int j = 0; j < loadedOpacity.size(); j++)
			{
				if (strcmp(loadedOpacity[j].path.data(), str.C_Str()) == 0)
				{
					texture.id = j;
					textures.push_back(texture);
					skip = true;
					break;
				}
			}
			break;
		}

		if (!skip)
		{
			//cout << typeName << endl;
			switch (typeName)
			{
			case DIFFUSE_MAP:
				texture.id = texID.idDiffuseMap;
				texID.idDiffuseMap++;
				break;
			case NORMAL_MAP:
				texture.id = texID.idNormalMap;
				texID.idNormalMap++;
				break;
			case OPACITY_MAP:
				texture.id = texID.idOpacityMap;
				texID.idOpacityMap++;
				break;
			}
			textures.push_back(texture);

			switch (typeName)
			{
			case DIFFUSE_MAP:
				loadedDiffuse.push_back(texture);
				break;
			case NORMAL_MAP:
				loadedNormal.push_back(texture);
				break;
			case OPACITY_MAP:
				loadedOpacity.push_back(texture);
				break;
			}
		}
	}

	return textures;
}

TexID Model::loadTexture(vector<Texture>& textures)
{
	TexID texID;
	texID.normalMap = UINT_MAX;
	texID.opacityMap = UINT_MAX;
	for (int k = 0; k < textures.size(); k++)
	{
		if (textures[k].type == NORMAL_MAP)
		{
			texID.normalMap = textures[k].id;
			if (flagLoadedNormal[textures[k].id])
				continue;

			int H_img, W_img, nrChannels;
			unsigned char* Imgdata = stbi_load(textures[k].path.c_str(), &W_img, &H_img, &nrChannels, 0);
			if (H_img == 0 || W_img == 0)
			{
				printf("Error: Texture path \"%s\" do not exist.\n", textures[k].path.c_str());
				free(Imgdata);
				//this->textures.erase(this->textures.begin() + k);
				continue;
			}
			else
			{
				Tex4f _normalMap;
				//cout << "Normal ID:" << textures[k].id << endl;
				_normalMap.data = (float4*)malloc(sizeof(float4) * H_img * W_img);
				_normalMap.H = H_img;
				_normalMap.W = W_img;

				int idx, idx_img;
				for (int i = 0; i < H_img; i++)
				{
					for (int j = 0; j < W_img; j++)
					{
						idx = i * W_img + j;
						idx_img = (i * W_img + j) * nrChannels;
						_normalMap.data[idx].x = ((float)Imgdata[idx_img + 0] / 255.f - 0.5f) * 2;
						_normalMap.data[idx].y = ((float)Imgdata[idx_img + 1] / 255.f - 0.5f) * 2;
						_normalMap.data[idx].z = ((float)Imgdata[idx_img + 2] / 255.f - 0.5f) * 2;
						_normalMap.data[idx].w = 1.f;
					}
				}
				this->texList.NormalMap.push_back(_normalMap);
				//showIntensity(this->texList.NormalMap[0].data, H_img, W_img);
				free(Imgdata);
				flagLoadedNormal[textures[k].id] = true;

			}
		}
		else if (textures[k].type == OPACITY_MAP)
		{
			texID.opacityMap = textures[k].id;
			if (flagLoadedOpacity[textures[k].id])
				continue;
			int H_img, W_img, nrChannels;
			unsigned char* Imgdata = stbi_load(textures[k].path.c_str(), &W_img, &H_img, &nrChannels, 0);
			if (H_img == 0 || W_img == 0)
			{
				printf("Error: Texture path \"%s\" do not exist.\n", textures[k].path.c_str());
				continue;
			}
			Tex1f _opacityMap;
			//cout << "Opacity ID:" << textures[k].id << endl;
			_opacityMap.data = (Pixel1f*)malloc(sizeof(Pixel1f) * H_img * W_img);
			_opacityMap.H = H_img;
			_opacityMap.W = W_img;
			int idx, idx_img;

			if (nrChannels == 4)
			{
				for (int i = 0; i < H_img; i++)
				{
					for (int j = 0; j < W_img; j++)
					{
						idx = i * W_img + j;
						idx_img = (i * W_img + j) * 4;
						_opacityMap.data[idx] = (float)Imgdata[idx_img + 3] / 255.f;
					}
				}
			}
			else if (nrChannels == 1)
			{
				for (int i = 0; i < H_img; i++)
				{
					for (int j = 0; j < W_img; j++)
					{
						idx = i * W_img + j;
						_opacityMap.data[idx] = (float)Imgdata[idx_img] / 255.f;
					}
				}
			}
			//this->opacity = 1.f;
			//showIdensity(NormalMap.data, H_img, W_img);
			this->texList.OpacityMap.push_back(_opacityMap);
			free(Imgdata);
			flagLoadedOpacity[textures[k].id] = true;
		}
		else if (textures[k].type == DIFFUSE_MAP)
		{
			texID.diffuseMap.push_back(textures[k].id);
			if (flagLoadedDiffuse[textures[k].id])
				continue;

			//cout << "Diffuse ID:" << textures[k].id << endl;
			int H_img, W_img, nrChannels;
			unsigned char* Imgdata = stbi_load(textures[k].path.c_str(), &W_img, &H_img, &nrChannels, 4);
			//cout << "Channel = " << nrChannels << " Path: " << textures[k].path.c_str() << "\n\n";
			if (H_img == 0 || W_img == 0)
			{
				printf("Error: Texture path \"%s\" do not exist.\n", textures[k].path.c_str());
				continue;
			}
			Tex4f tmp;
			tmp.H = H_img;
			tmp.W = W_img;
			tmp.data = (float4*)malloc(sizeof(float4) * H_img * W_img);
			int idx, idx_img;

			for (int i = 0; i < H_img; i++)
			{
				for (int j = 0; j < W_img; j++)
				{
					idx = i * W_img + j;

					idx_img = idx * 4;
					float4 t = { (float)Imgdata[idx_img], (float)Imgdata[idx_img + 1], (float)Imgdata[idx_img + 2],  (float)Imgdata[idx_img + 3] / 255. };
					tmp.data[idx] = t;
				}
			}

			free(Imgdata);
			this->texList.DiffuseMap.push_back(tmp);
			flagLoadedDiffuse[textures[k].id] = true;
		}
	}
	return texID;
}