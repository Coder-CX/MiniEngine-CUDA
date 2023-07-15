# miniEngine_CUDA 

本项目为基于CUDA的可编程渲染管线，模仿了OpenGL和DirectX的基础功能

顶点着色器在CPU端计算，片元着色器在GPU端渲染，效率相较纯CPU软着色器（RenderHelp）效率更高

## 参考项目

LearnOpenGL: https://learnopengl.com/

RenderHelp: https://github.com/skywind3000/RenderHelp

## 依赖库

CUDA版本：12.1

图像读取/写入：STB_IMAGE：https://github.com/nothings/stb

模型读取：Assimp： https://github.com/assimp/assimp

数学：Eigen： http://www.eigen.tuxfamily.org/index.php?title=Main_Page

代码编写环境：CPU：i9-13900HX, GPU：RTX 4060, RAM：16G

IDE：Visual Studio 2022

## 可实现的功能

1. 基础渲染：输入顶点数组、顶点序号数组，实现基础渲染

2. 模型读取：使用Model类管理模型，可读取多种3D文件的顶点，顶点编号、贴图纹理信息（基于Assimp）并转移至显存（VRAM）

3. 顶点着色器与片元着色器可编程（使用CUDA extended lambda与模板函数实现函数传输）

4. 自定义绘制帧，最多同时存储帧数量由MAX_FRAMEBUFFER_NUM宏定义（Basic.h），仅在添加时分配内存

5. 绑定帧为贴图，可进行图像的后期处理与采样，可自定义贴图读取信息（使用cuda纹理对象读取，纹理读取信息由cudaTextureDesc结构体定义）

6. 自定义深度缓冲（已实现）与模板缓冲（待实现），可实现阴影贴图计算

7. 图片输出为PNG格式，可使用其他图像软件（如mspaint.exe）打开
