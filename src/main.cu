#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

const char * WINDOW_TITLE = "RayCaster - Cuda";

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "gl_util.h"
#include "raycaster.h"

#define WALK_SPEED 3.0f
#define CAM_SPEED 1.5f

// -------------------------------------
// MARK: WINDOW SETUP & LIFETIME METHODS
// -------------------------------------

int main() {
	caster_setup(WINDOW_TITLE);
	caster_load_assets("tex/sheet.png", "tex/skybox.png", "tex/lava.jpg");

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
			camrot -= time_delta * CAM_SPEED;
		} else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			camrot += time_delta * CAM_SPEED;
		}

		caster_update(posx, posy, camrot);
	}

	// Done - cleanup
	caster_cleanup();
	glfwTerminate();
	return 0;
}
