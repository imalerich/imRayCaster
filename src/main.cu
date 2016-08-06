#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "gl_util.hpp"

const char * WINDOW_TITLE = "RayCaster - Cuda";
void present_gl();

surface<void, 2> tex;
__global__ void runCuda(float4 * tex, int screen_w, int screen_h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < screen_w && y < screen_h) {
		float val = x / (float)screen_w;
		tex[y + screen_w + y] = make_float4(val, 1.0f, 0.0f, 1.0f);
	}
}

void check_err(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
	} else {
		printf("CUDA returned success.\n");
	}
}

int main() {
	init_gl(WINDOW_TITLE, VSYNC_ENABLED);

	struct cudaGraphicsResource * tex_res;
	cudaGraphicsGLRegisterImage(&tex_res, screen_tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
	float4 * d_tex;
	cudaGraphicsMapResources(1, &tex_res, 0);
	size_t num_bytes;
	check_err(cudaGraphicsResourceGetMappedPointer((void **)&d_tex, &num_bytes, tex_res));

	dim3 block(16, 16);
	dim3 grid((screen_w + block.x - 1) / block.x,
			  (screen_h + block.y - 1) / block.y);
	runCuda<<<grid, block>>>(d_tex, screen_w, screen_h);
	cudaGraphicsUnmapResources(1, &tex_res, 0);

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
