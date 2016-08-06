#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudaGL.h>
#include <stdio.h>
#include <stdlib.h>

#include "gl_util.hpp"

const char * WINDOW_TITLE = "RayCaster - Cuda";
void present_gl();

// This is the actual pointer we will operate on.
CUarray cu_arr;

void init_cuda() {
	// CUgraphicsResource is a temporary link from the GL texture to the CUDA object.
	CUgraphicsResource screen_resource;
	cuGraphicsGLRegisterImage(
			&screen_resource,
			screen_tex,
			GL_TEXTURE_2D,
			CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE
			);
	cuGraphicsMapResources(1, &screen_resource, 0);
	cuGraphicsSubResourceGetMappedArray(&cu_arr, screen_resource, 0, 0);
	cuGraphicsUnmapResources(1, &screen_resource, 0);
}

surface<void, 2> tex;
__global__ void runCuda(int screen_w, int screen_h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < screen_w && y < screen_h) {
		unsigned char val = 255 * x / (float)screen_w;
		uchar4 tmp = {val, val, val, 0};
		surf2Dwrite(tmp, tex, x * sizeof(float), y);
	}
}

int main() {
	init_gl(WINDOW_TITLE, VSYNC_ENABLED);
	init_cuda();

	cuModuleLoad();
	cuModuleGetSurfRef(&surf, m_module, "tex");

	cuTexRefSetArray(, cu_arr, CU_TRSA_OVERRIDE_FORMAT);
	dim3 block(16, 16);
	dim3 grid((screen_w + block.x - 1) / block.x,
			  (screen_h + block.y - 1) / block.y);
	runCuda<<<grid, block>>>(d_tex, screen_w, screen_h);

	// Game loop.
	glfwSetTime(0.0f);
	while (!glfwWindowShouldClose(window)) {
		// Close on escape press.
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}

		// TODO

		present_gl();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Done - cleanup
	glfwTerminate();
	return 0;
}

/**
 * Push a new frame to the screen.
 * This will contain the 'screen_tex' managed by gl_util.
 */
void present_gl() {
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	update_screen();
}
