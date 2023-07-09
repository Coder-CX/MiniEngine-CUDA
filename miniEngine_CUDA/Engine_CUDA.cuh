#pragma once

#include "Basic.h"
#include <device_launch_parameters.h>

void fillPixel1f(float* data, float color, int H, int W);
void fillPixel4f(float4* data, float4 color, int H, int W);
void convertTex4f2Tex4i(uchar4* dst, float4* src, int H, int W);
