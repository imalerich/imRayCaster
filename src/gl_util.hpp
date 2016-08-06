//
//  gl_util.h
//  OpenCL-Mac
//
//  Created by Ian Malerich on 1/21/16.
//  Copyright © 2016 Ian Malerich. All rights reserved.
//

#ifndef GL_UTIL_H
#define GL_UTIL_H

#define GLEW_STATIC
#include <GL/glew.h>

#define VSYNC_DISABLED 0
#define VSYNC_ENABLED 1

#include <GLFW/glfw3.h>

#ifdef __APPLE__
	#define GLFW_EXPOSE_NATIVE_COCOA
	#define GLFW_EXPOSE_NATIVE_NSGL
	#include <OpenGL/OpenGL.h>
#elif defined __linux__
	#define GLFW_EXPOSE_NATIVE_X11
	#define GLFW_EXPOSE_NATIVE_GLX
	#include <GL/glx.h>
#elif defined __MINGW32__
	#define GLFW_EXPOSE_NATIVE_WIN32	
	#define GLFW_EXPOSE_NATIVE_WGL
	#include <windows.h>
	#include <Wingdi.h>
#endif

#include <GLFW/glfw3native.h>

extern GLFWwindow * window;
extern unsigned screen_w;
extern unsigned screen_h;

// make the frame rate available
extern float last_time;
extern float current_time;
extern float time_passed;
extern float fps;

/**
 serves as a multiplier to the screen_w and the screen_h
 when generating the screen_tex, the resulting pixel count
 will be screen_w * screen_h * sample_rate^2
 the default value of sample_rate is 1
 */
extern unsigned sample_rate;
extern GLuint screen_tex;

/**
 Initializes an OpenGL context using screen 
 dimmensions of screen_w and screen_h.
 \param title Title to use for the window.
 \param v_sync The swap interval, use 0 to disable vertical sync.
 */
void init_gl(const char * title, int v_sync);

/**
 Draws a rectangle over the entire screen using the
 texture to be filled by OpenCL.
 */
void update_screen();

/**
 Checks if an operation has produced an error since last
 checking for an error. If so, this method will print out
 the information detailed in 'info' as well as the error code
 to stderr, and then exit with a failure.
 */
void gl_check_errors(const char * info);

#endif
