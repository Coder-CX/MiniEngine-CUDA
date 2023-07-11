#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <vector_types.h>
#include <cuda_runtime.h>

#define PI 3.14159
#define MAX_CONTEXT_SIZE 4
#define MAX_TRIANGLE_BUFFER 512
#define MAX_FRAMEBUFFER_NUM 16

#define BDIM_X 32
#define BDIM_Y 32


using std::vector;
using std::string;

using vec2 = Eigen::Vector2f;
using vec3 = Eigen::Vector3f;
using vec4 = Eigen::Vector4f;
using vec2i = Eigen::Vector2i;
using vec3i = Eigen::Vector3i;
using vec4i = Eigen::Vector4i;

using mat2 = Eigen::Matrix2f;
using mat3 = Eigen::Matrix3f;
using mat4 = Eigen::Matrix4f;
using mat2i = Eigen::Matrix2i;
using mat3i = Eigen::Matrix3i;
using mat4i = Eigen::Matrix4i;

enum lightType { POINT_LIGHT, SUN_LIGHT, CONE_LIGHT };
enum interpolationType { NEAREST, BILINEAR };
enum textureType { DIFFUSE_MAP, NORMAL_MAP, SPECULAR_MAP, HEIGHT_MAP, DEPTH_MAP, SHADOW_MAP, OPACITY_MAP };

typedef struct Context
{
	float* vec1f;
	float2* vec2f;
	float3* vec3f;
	float4* vec4f;
} Context;

typedef struct ContextInside
{
	float vec1f[4];
	float2 vec2f[4];
	float3 vec3f[4];
	float4 vec4f[4];
} ContextInside;

typedef struct VertexShader
{
	Context context;
	float rhw;
	vec4 pos;
	vec2 pos_sf;
	vec2i pos_si;
} Vertex_S;

typedef struct Box
{
	int min_X, max_X;
	int min_Y, max_Y;
} Box;

typedef struct Tex_Shader
{
	int H, W;
	cudaTextureObject_t tex;
	cudaArray_t tex_data;
} Tex_Shader;