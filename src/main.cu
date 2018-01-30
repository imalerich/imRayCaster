#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "gl_util.h"

// ------------------
// MARK: DECLARATIONS
// ------------------

#define M_EPSILON 0.000001f
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
	1, 1, 1, 1, 1, 1, 2, 2, 2, 1,
	1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
	1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 2, 0, 0, 0, 2, 0, 0, 1,
	2, 0, 0, 1, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 1, 0, 2,
	2, 0, 2, 0, 0, 0, 0, 1, 0, 2,
	1, 0, 2, 2, 0, 0, 0, 1, 0, 2,
	2, 0, 0, 0, 0, 0, 0, 0, 0, 1,
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

/** Computes the texture coordinate for the input position. */
__device__ float2 calc_tex_coord(float2 pos, unsigned y, unsigned screen_h, float H) {
	float x_dist = min(ceil(pos.x) - pos.x, pos.x - floor(pos.x));
	float y_dist = min(ceil(pos.y) - pos.y, pos.y - floor(pos.y));

	float u = x_dist > y_dist ? pos.x - floor(pos.x) : pos.y - floor(pos.y);
	float v = (y - (screen_h - H) * 0.5f) / H;

	return make_float2(u, v);
}

/** Computes texture coordinates for either the floor or ceiling. */
__device__ float2 calc_base_tex_coord(float y, unsigned screen_h, float ROT, float cam_rot, float2 P) {
	// how tall would a wall have to be
	// if the bottom would be on this layer of floor?
	const float H = 2.0f * abs(y - screen_h * 0.5);
	// at what distance is the base of that wall
	// relative to the camera
	const float d = (screen_h / H);

	// tiles repeat, so texture coordinates are 
	// the world coordinates divided by the wall size
	float2 uv;

	uv.x = d * tan(ROT);
	uv.y = d;

	// transform the coordinate relative to the camera
	uv = rotate(uv, -cam_rot);
	uv.x -= P.x; uv.y += P.y;

	return uv;
}

surface<void, 2> tex;
texture<float4, 2, cudaReadModeElementType> wall1;
texture<float4, 2, cudaReadModeElementType> wall2;
texture<float4, 2, cudaReadModeElementType> ground;
texture<float4, 2, cudaReadModeElementType> roof;

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
		x_dist += (look.x > 0.0f ? M_EPSILON : -M_EPSILON);
		y_dist += (look.y > 0.0f ? M_EPSILON : -M_EPSILON);

		// how 'long' will it take to reach that wall?
		float tx = abs(x_dist / look.x);
		float ty = abs(y_dist / look.y);
		float t = min(tx, ty);

		// move to the nearest wall using the look vector
		pos.x += t * look.x;
		pos.y += t * look.y;
	}

	float2 HIT = pos; // actual hit position, pre normalization

	const float d = dist(pos, P) * cos(ROT);

	// the height (in pixels) of the wall we hit
	const float H = screen_h / d;
	uchar4 data;

	// check if the current y position should render for the hit wall height
	if (y > (screen_h - H) * 0.5f && y < (screen_h + H) * 0.5f) {
		// compute the normal vector of the hit surface for lighting
		float2 norm = calc_map_norm(HIT);
		float2 uv = calc_tex_coord(HIT, y, screen_h, H);

		const float2 LIGHT = normalize(make_float2(1.0f, 1.0f));
		float s = max(dot(norm, LIGHT), 0.5f);

		float4 c = sample_map(HIT) == 1 ?
			tex2D(wall1, uv.x, uv.y) :
			tex2D(wall2, uv.x, uv.y);
		data = make_color(s * c.x, s * c.y, s * c.z);
	} else if (y > screen_h * 0.5) {
		float2 uv = calc_base_tex_coord(y, screen_h, ROT, cam_rot, P);
		float4 c = tex2D(ground, uv.x, uv.y);
		data = make_color(c.x, c.y, c.z);
	} else {
		float2 uv = calc_base_tex_coord(y, screen_h, ROT, cam_rot, P);
		float4 c = tex2D(roof, uv.x, uv.y);
		data = make_color(c.x, c.y, c.z);
	}

	surf2Dwrite<uchar4>(data, tex, x * sizeof(uchar4), y);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
void loadTexForCuda(struct texture<T, dim, readMode> &tex, struct cudaArray * &arr, const char * filename) {

	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	int channels, width, height;
	unsigned char * udata = stbi_load(filename, &width, &height, &channels, 4);
	unsigned size = width * height * 4 * sizeof(float);

	// convert the unsigned data into floating point data
	float * data = (float *)malloc(size);
	for (int i=0; i<width*height*4; i++) { data[i] = udata[i] / 255.0f; }

	cudaMallocArray(&arr, &channelDesc, width, height);
	cudaMemcpyToArray(arr, 0, 0, data, size, cudaMemcpyHostToDevice);

	// don't need the device memory anymore, release it
	free(udata);
	free(data);

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = true;

	cudaBindTextureToArray(tex, arr, channelDesc);
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

	/* --- Setup the Wall Texture for Rendering. --- */

	struct cudaArray * wall1_arr = 0;
	struct cudaArray * wall2_arr = 0;
	struct cudaArray * ground_arr = 0;
	struct cudaArray * roof_arr = 0;

	loadTexForCuda(wall1, wall1_arr, "tex/wall0.png");
	loadTexForCuda(wall2, wall2_arr, "tex/wall4.png");
	loadTexForCuda(ground, ground_arr, "tex/ground0.png");
	loadTexForCuda(roof, roof_arr, "tex/wall1.png");

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
