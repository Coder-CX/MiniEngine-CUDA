#include <iostream>
#include "Model.h"
#include "Engine.h"

int main(void)
{
	Engine scene(1920, 1080);
	scene.addFrame4f();
	float4 color = { 1. ,0., 0., 1. };
	scene.fill4f(0, color);
	scene.saveFrame4f(0, "test.png");
	system("mspaint.exe test.png");
	return 0;
}