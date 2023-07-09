#pragma once

#include "Basic.h"
#include "Camera.h"

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
	float const tanHalfFovy = tan(fovy / 2);
	mat4 Result = mat4::Zero();
	Result(0, 0) = 1.f / (aspect * tanHalfFovy);
	Result(1, 1) = 1.f / (tanHalfFovy);
	Result(2, 2) = (zFar + zNear) / (zNear - zFar);
	Result(3, 2) = 1.f;
	Result(2, 3) = (2 * zFar * zNear) / (zFar - zNear);
	return Result;
}

mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
{
	mat4 Result = mat4::Identity();
	Result(0, 0) = -2. / (right - left);
	Result(1, 1) = -2. / (top - bottom);
	Result(2, 2) = 2. / (zNear - zFar);
	Result(0, 3) = (right + left) / (right - left);
	Result(1, 3) = (top + bottom) / (top - bottom);
	Result(2, 3) = -(zFar + zNear) / (zNear - zFar);
	return Result;

}

mat4 lookAt(Camera cam)
{
	vec3 camDir = (cam.pos - cam.target).normalized();
	vec3 camRight = (cam.up.cross(camDir)).normalized();
	vec3 camUp = camDir.cross(camRight);

	mat4 Result = mat4::Identity();
	Result.row(0).head(3) = camRight.transpose();
	Result.row(1).head(3) = camUp.transpose();
	Result.row(2).head(3) = camDir.transpose();
	mat4 shift = mat4::Identity();
	shift.col(3).head(3) = -cam.pos;

	return Result * shift;
}