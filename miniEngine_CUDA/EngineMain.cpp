#include <iostream>
#include "Model.h"

int main(void)
{
	Model obj("D:\\Code\\C\\HoloGen\\res\\shadow_rabbit_200mm.obj");
	printf("%d\n", obj.texList_d.diffuse_CUDA_HW[0].x);
	return 0;
}