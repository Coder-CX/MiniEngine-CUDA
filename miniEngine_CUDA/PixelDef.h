#pragma once

#include "Basic.h"

typedef float Pixel1f;

typedef struct {
	float x;
	float y;
} Pixel2f;

typedef struct {
	float x;
	float y;
	float z;
} Pixel3f;

typedef struct {
	float x;
	float y;
	float z;
	float w;
} Pixel4f;

static Pixel2f operator+(const Pixel2f& a, const Pixel2f& b)
{
	Pixel2f _t;
	_t.x = a.x + b.x;
	_t.y = a.y + b.y;
	return _t;
}

static Pixel3f operator+(const Pixel3f& a, const Pixel3f& b)
{
	Pixel3f _t;
	_t.x = a.x + b.x;
	_t.y = a.y + b.y;
	_t.z = a.z + b.z;
	return _t;
}

static Pixel4f operator+(const Pixel4f& a, const Pixel4f& b)
{
	Pixel4f _t;
	_t.x = a.x + b.x;
	_t.y = a.y + b.y;
	_t.z = a.z + b.z;
	_t.w = a.w + b.w;
	return _t;
}

static Pixel2f operator*(const Pixel2f& a, const float b)
{
	Pixel2f _t;
	_t.x = a.x * b;
	_t.y = a.y * b;
	return _t;
}

static Pixel3f operator*(const Pixel3f& a, const float b)
{
	Pixel3f _t;
	_t.x = a.x * b;
	_t.y = a.y * b;
	_t.z = a.z * b;
	return _t;
}

static Pixel4f operator*(const Pixel4f& a, const float b)
{
	Pixel4f _t;
	_t.x = a.x * b;
	_t.y = a.y * b;
	_t.z = a.z * b;
	_t.w = a.w * b;
	return _t;
}

static inline vec3 Pixel3f2Vec3(Pixel3f&& in)
{
	return vec3(in.x, in.y, in.z);
}

static inline vec3 Pixel4f2Vec3(Pixel4f&& in)
{
	return vec3(in.x, in.y, in.z);
}

static inline float RGB2Gray(Pixel3f p)
{
	return p.x * 0.2126 + p.y * 0.7152 + p.z * 0.0722;
}

static inline float RGBA2Gray(Pixel4f p)
{
	return (p.x * 0.2126 + p.y * 0.7152 + p.z * 0.0722) * p.w;
}