#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gl_util.h"

// ------------------
// MARK: DECLARATIONS
// ------------------

#define M_EPSILON 0.00001f
#define DEGREES_TO_RAD(deg) ((deg / 180.0f) * M_PI)

#define WALK_SPEED 2.0f
#define MAX_ITER 100
#define WALL_SIZE 1.0f
#define FOV_DEGREES 90.0f
#define FOV DEGREES_TO_RAD(FOV_DEGREES)
#define DEPTH_FACTOR 5.0f

#define MAP_WIDTH 10
#define MAP_HEIGHT 10

__device__ int MAP[] = {
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
	1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
	1, 0, 1, 0, 0, 0, 0, 1, 0, 1,
	1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

const char * WINDOW_TITLE = "RayCaster - Cuda";
void present_gl();

// -----------------
// MARK: DEVICE CODE
// -----------------

/** Given RGB input on a [0.0,1.0] scale, create a color which we can output.  */
__device__ uchar4 make_color(float r, float g, float b) {
	return make_uchar4(r * 255, g * 255, b * 255, 255);
}

/** Computes the magnitude of the input vector. */
__device__ float mag(float2 v) {
	return sqrt(v.x * v.x + v.y * v.y);
}

/** Normalize the input vector. */
__device__ float2 normalize(float2 v) {
	const float M = mag(v);
	return make_float2(v.x / M, v.y / M);
}

/** Compute the vector dot product of the two input vectors. */
__device__ float dot(float2 v0, float2 v1) {
	return v0.x * v1.x + v0.y * v1.y;
}

/** Compute the distance between two vectors. */
__device__ float dist(float2 v0, float2 v1) {
	return sqrt(pow(v0.x - v1.x, 2) + pow(v0.y - v1.y, 2));
}

/** Rotate the input vector by the given angle 'r' in radians. */
__device__ float2 rotate(float2 v, float r) {
	return make_float2(
		dot(v, make_float2(cos(r), -sin(r))),
		dot(v, make_float2(sin(r), cos(r)))
	);
}

/** Sample the MAP[] array for the given position. */
__device__ int sample_map(float2 pos) {
	int x = (int)(pos.x / WALL_SIZE);
	int y = (int)(pos.y / WALL_SIZE);

	if (x >= MAP_WIDTH || x < 0 || y >= MAP_HEIGHT || y < 0) { return 1; }

	return MAP[MAP_WIDTH * y + x];
}

/** Computes the normal vector associated with the block relative to the given intersection pos. */
__device__ float2 calc_map_norm(float2 pos) {
	float x_dist = min(ceil(pos.x) - pos.x, pos.x - floor(pos.x));
	float y_dist = min(ceil(pos.y) - pos.y, pos.y - floor(pos.y));

	float x = 0.0f, y = 0.0f;
	if (x_dist < y_dist) {
		x = (ceil(pos.x) - pos.x < pos.x - floor(pos.x)) ? -1.0f : 1.0f;
	} else {
		y = (ceil(pos.y) - pos.y < pos.y - floor(pos.y)) ? -1.0f : 1.0f;
	}

	return make_float2(x, y);
}

surface<void, 2> tex;
__global__ void runCuda(float posx, float posy, float cam_rot, unsigned screen_w, unsigned screen_h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// don't do any off screen work
	// and certainly don't write off the texture buffer
	if (x >= screen_w || y >= screen_h) { return; }

	const float2 P = make_float2(posx, posy);
	const float2 L = rotate(normalize(make_float2(0.0f, 1.0f)), cam_rot);

	// current x position ranging from -1.0 to 1.0
	const float XPERC = -2.0f * (x / (float)screen_w) + 1.0;
	const float ROT = FOV * 0.5f * XPERC;

	float2 look = rotate(L, ROT);

	float2 pos = P;
	for (int iter = 0; iter < MAX_ITER && sample_map(pos) == 0; iter++) {
		// distance to the nearest wall on each dimension
		float x_dist = (look.x > 0.0f ? ceil(pos.x) : floor(pos.x)) - pos.x;
		float y_dist = (look.y > 0.0f ? ceil(pos.y) : floor(pos.y)) - pos.y;

		// move a smidge more than necessary to guarantee 
		// we actually moved to a new region
		x_dist += (look.x > 0.0f ? M_EPSILON : -M_EPSILON) * WALL_SIZE;
		y_dist += (look.y > 0.0f ? M_EPSILON : -M_EPSILON) * WALL_SIZE;

		// how 'long' will it take to reach that wall?
		float tx = abs(x_dist / look.x);
		float ty = abs(y_dist / look.y);
		float t = min(tx, ty);

		// move to the nearest wall using the look vector
		pos.x += t * look.x;
		pos.y += t * look.y;
	}

	// compute the normal vector of the hit surface for lighting
	float2 norm = calc_map_norm(pos);

	// adjust pos relative to the camera looking forward
	pos.x -= P.x; pos.y -= P.y;
	pos = rotate(pos, -cam_rot);

	const float N = 0.1f;
	const float F = 100.0f;
	const float d = pos.y * ((F + N)/(F-N)) + ((2*N*F)/(F-N));

	// the height (in pixels) of the wall we hit
	const float H = screen_h / d;

	// float g = (d-1.5f); /* debug greyscale output */
	uchar4 data = make_color(137/255.0f, 137/255.0f, 137/255.0f);

	// check if the current y position should render for the hit wall height
	if (y > (screen_h - H) * 0.5f && y < (screen_h + H) * 0.5f) {

		const float2 LIGHT = normalize(make_float2(1.0f, 1.0f));
		float s = max(dot(norm, LIGHT), 0.5f);
		data = make_color(s, s, s);
	} else if (y < screen_h / 2) {
		data = make_color(46/255.0f, 47/255.0f, 48/255.0f);
	}

	surf2Dwrite<uchar4>(data, tex, x * sizeof(uchar4), y);
}

// -------------------------------------
// MARK: WINDOW SETUP & LIFETIME METHODS
// -------------------------------------

void check_err(cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(err));
		exit(0);
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

	float posx = MAP_WIDTH * 0.5f * WALL_SIZE;
	float posy = MAP_HEIGHT * 0.5f * WALL_SIZE;
	float camrot = 0.0f;

	// Game loop.
	glfwSetTime(0.0f);
	float last_time = 0.0f;
	while (!glfwWindowShouldClose(window)) {
		float time_delta = glfwGetTime() - last_time;
		last_time = glfwGetTime();

		// Close on escape press.
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			float xdir = -sin(camrot) * time_delta;
			float ydir = cos(camrot) * time_delta;
			posx += xdir * WALK_SPEED;
			posy += ydir * WALK_SPEED;
		} else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			float xdir = -sin(camrot) * time_delta;
			float ydir = cos(camrot) * time_delta;
			posx -= xdir * WALK_SPEED;
			posy -= ydir * WALK_SPEED;
		}

		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
			float xdir = cos(camrot) * time_delta;
			float ydir = sin(camrot) * time_delta;
			posx -= xdir * WALK_SPEED;
			posy -= ydir * WALK_SPEED;
		} else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
			float xdir = cos(camrot) * time_delta;
			float ydir = sin(camrot) * time_delta;
			posx += xdir * WALK_SPEED;
			posy += ydir * WALK_SPEED;
		}

		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			camrot -= time_delta;
		} else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			camrot += time_delta;
		}

		// Run our CUDA kernel to generate the image.
		dim3 block(8, 8);
		dim3 grid((screen_w + block.x - 1) / block.x,
				  (screen_h + block.y - 1) / block.y);
		runCuda<<<grid, block>>>(posx, posy, camrot, screen_w, screen_h);
		cudaStreamSynchronize(0);

		cudaGraphicsUnmapResources(1, &tex_res, 0);

		present_gl();
		glfwSwapBuffers(window);
		glfwPollEvents();

		cudaGraphicsMapResources(1, &tex_res, 0);
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
