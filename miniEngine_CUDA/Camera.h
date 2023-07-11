#pragma once

#include "Basic.h"

class Camera
{
public:
	vec3 pos;
	vec3 target;
	vec3 up;
	float FOV;

	Camera(vec3 pos, vec3 target, vec3 up, float FOV = PI / 2)
	{
		this->pos = pos;
		this->target = target;
		this->up = up;
		this->FOV = FOV;
	}
};