#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#define PI 3.14159
#define MAX_CONTEXT_SIZE 4
#define MAX_CONTEXT_BUFFER 1024
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