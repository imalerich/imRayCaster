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
__global__ void runCuda(float time, unsigned screen_w, unsigned screen_h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < screen_w && y < screen_h) {
		float val = time * (y / (float)screen_h) * (x / (float)screen_w);
		uchar4 data = make_uchar4(val * 121, val * 212, val * 175, 255);
		surf2Dwrite<uchar4>(data, tex, x * sizeof(uchar4), y);
	}
}

void check_err(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
		exit(0);
	} else {
		printf("CUDA returned success.\n");
	}
}

int main() {
	init_gl(WINDOW_TITLE, VSYNC_ENABLED);

	struct cudaGraphicsResource * tex_res;
	struct cudaArray * cu_arr;

	cudaSetDevice(0);
	cudaGLSetGLDevice(0);
	cudaGraphicsGLRegisterImage(&tex_res, screen_tex, GL_TEXTURE_2D, 
			cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &tex_res, 0);
	cudaGraphicsSubResourceGetMappedArray(&cu_arr, tex_res, 0, 0);
	cudaBindSurfaceToArray(tex, cu_arr);

	// Game loop.
	glfwSetTime(0.0f);
	while (!glfwWindowShouldClose(window)) {
		// Close on escape press.
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}

		// Run our CUDA kernel to generate the image.
		dim3 block(8, 8);
		dim3 grid((screen_w + block.x - 1) / block.x,
				  (screen_h + block.y - 1) / block.y);
		float time = glfwGetTime() / 5.0f;
		runCuda<<<grid, block>>>(time, screen_w, screen_h);
		cudaGraphicsUnmapResources(1, &tex_res, 0);
		cudaStreamSynchronize(0);

		present_gl();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Done - cleanup
	cudaGraphicsUnregisterResource(tex_res);
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
