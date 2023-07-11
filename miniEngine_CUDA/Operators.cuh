#pragma once

#include "Basic.h"
#include <device_launch_parameters.h>

__host__ __device__
inline float2 operator+(const float2& a, const float2& b)
{
	return { a.x + b.x, a.y + b.y };
}

__host__ __device__
inline float2 operator+(const float2& a, const float& b)
{
	return { a.x + b, a.y + b };
}

__host__ __device__
inline float2& operator+=(float2& a, const float2& b)
{
	a = a + b;
	return a;
}

__host__ __device__
inline float2& operator+=(float2& a, const float& b)
{
	a = { a.x + b, a.y + b };
	return a;
}

__host__ __device__
inline float2 operator-(const float2& a, const float2& b)
{
	return { a.x - b.x, a.y - b.y };
}

__host__ __device__
inline float2 operator-(const float2& a, const float& b)
{
	return { a.x - b, a.y - b };
}

__host__ __device__
inline float2& operator-=(float2& a, const float2& b)
{
	a = a - b;
	return a;
}

__host__ __device__
inline float2& operator-=(float2& a, const float& b)
{
	a = { a.x - b, a.y - b };
	return a;
}


__host__ __device__
inline float2 operator*(const float2& a, const float& b)
{
	return { a.x * b, a.y * b };
}

__host__ __device__
inline float2 operator*(const float& a, const float2& b)
{
	return { a * b.x, a * b.y };
}

__host__ __device__
inline float2& operator*=(float2& a, const float& b)
{
	a = { a.x * b, a.y * b };
	return a;
}

__host__ __device__
inline float2 operator/(const float2& a, const float& b)
{
	return { a.x / b, a.y / b };
}

__host__ __device__
inline float2& operator/=(float2& a, const float& b)
{
	a = { a.x / b, a.y / b };
	return a;
}

__host__ __device__
inline float3 operator*(const float3& a, const float& b)
{
	return { a.x * b, a.y * b, a.z * b };
}

__host__ __device__
inline float3 operator*(const float& a, const float3& b)
{
	return { a * b.x, a * b.y, a * b.z };
}

__host__ __device__
inline float3& operator*=(float3& a, const float& b)
{
	a = a * b;
	return a;
}

__host__ __device__
inline float3 operator/(const float3& a, const float& b)
{
	return { a.x / b, a.y / b, a.z / b };
}

__host__ __device__
inline float3& operator/=(float3& a, const float& b)
{
	a = a / b;
	return a;
}

__host__ __device__
inline float4 operator*(const float4& a, const float& b)
{
	return { a.x * b, a.y * b, a.z * b, a.w };
}

__host__ __device__
inline float4 operator*(const float& a, const float4& b)
{
	return { a * b.x, a * b.y, a * b.z, a * b.w };
}

__host__ __device__
inline float4& operator*=(float4& a, const float& b)
{
	a = a * b;
	return a;
}

__host__ __device__
inline float4 operator/(const float4& a, const float& b)
{
	return { a.x / b, a.y / b, a.z / b, a.w / b };
}

__host__ __device__
inline float4& operator/=(float4& a, const float& b)
{
	a = a / b;
	return a;
}


__host__ __device__
inline float3 operator+(const float3& a, const float3& b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__
inline float3 operator+(const float3& a, const float& b)
{
	return { a.x + b, a.y + b, a.z + b };
}

__host__ __device__
inline float3& operator+=(float3& a, const float3& b)
{
	a = a + b;
	return a;
}

__host__ __device__
inline float3& operator+=(float3& a, const float& b)
{
	a = a + b;
	return a;
}

__host__ __device__
inline float3 operator-(const float3& a, const float3& b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__
inline float3 operator-(const float3& a, const float& b)
{
	return { a.x - b, a.y - b, a.z - b };
}

__host__ __device__
inline float3& operator-=(float3& a, const float3& b)
{
	a = a - b;
	return a;
}

__host__ __device__
inline float3& operator-=(float3& a, const float& b)
{
	a = a - b;
	return a;
}

__host__ __device__
inline float4 operator+(const float4& a, const float4& b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

__host__ __device__
inline float4 operator+(const float4& a, const float& b)
{
	return { a.x + b, a.y + b, a.z + b, a.w + b };
}

__host__ __device__
inline float4& operator+=(float4& a, const float4& b)
{
	a = a + b;
	return a;
}

__host__ __device__
inline float4& operator+=(float4& a, const float& b)
{
	a = a + b;
	return a;
}

__host__ __device__
inline float4 operator-(const float4& a, const float4& b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

__host__ __device__
inline float4 operator-(const float4& a, const float& b)
{
	return { a.x - b, a.y - b, a.z - b, a.w - b };
}

__host__ __device__
inline float4& operator-=(float4& a, const float4& b)
{
	a = a - b;
	return a;
}

__host__ __device__
inline float4& operator-=(float4& a, const float& b)
{
	a = a - b;
	return a;
}
