#pragma once

#include "Basic.h"
#include "Camera.h"

__host__ __device__
inline vec4 vecXYZ1(vec3& v)
{
	return vec4(v(0), v(1), v(2), 1.f);
}

__host__ __device__
inline float2 vec2ToFloat2(vec2& v)
{
	return { v(0), v(1) };
}

__host__ __device__
inline float3 vec3ToFloat3(vec3& v)
{
	return { v(0), v(1), v(2)};
}

__host__ __device__
inline vec2 float2ToVec2(float2& v)
{
	return vec2(v.x, v.y);
}

__host__ __device__
inline vec3 float3ToVec3(float3 & v)
{
	return vec3(v.x, v.y, v.z);
}

mat4 view(int H, int W)
{
	mat4 result = mat4::Identity();
	result(0, 0) = W >> 1;
	result(1, 1) = H >> 1;
	result(0, 3) = (W - 1) / 2;
	result(1, 3) = (H - 1) / 2;
	return result;
}

mat4 perspective(float fovy, float aspect, float zNear, float zFar)
{
	float const tanHalfFovy = tan(fovy * .5f);
	mat4 Result = mat4::Zero();
	Result(0, 0) = 1.f / (aspect * tanHalfFovy);
	Result(1, 1) = 1.f / (tanHalfFovy);
	Result(2, 2) = (zFar) / (zFar - zNear);
	Result(3, 2) = 1.f;
	Result(2, 3) = - zFar * zNear / (zFar - zNear);
	return Result;
}

mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
{
	mat4 Result = mat4::Identity();
	Result(0, 0) = 2. / (right - left);
	Result(1, 1) = 2. / (top - bottom);
	Result(2, 2) = 2. / (zNear - zFar);
	Result(0, 3) = (right + left) / (right - left);
	Result(1, 3) = (top + bottom) / (top - bottom);
	Result(2, 3) = -(zFar + zNear) / (zNear - zFar);
	return Result;

}

mat4 lookAt(Camera cam)
{
	vec3 camDir = (cam.target - cam.pos).normalized();
	vec3 camRight = (cam.up.cross(camDir)).normalized();
	vec3 camUp = camDir.cross(camRight);

	mat4 Result = mat4::Identity();
	Result.row(0).head(3) = camRight;
	Result.row(1).head(3) = camUp;
	Result.row(2).head(3) = camDir;
	Result.col(3).head(3) = vec3(-cam.pos.dot(camRight), -cam.pos.dot(camUp), -cam.pos.dot(camDir));
	return Result;
}