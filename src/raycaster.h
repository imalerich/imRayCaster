#ifndef __RAYCASTER_H__
#define __RAYCASTER_H__

#include <string>
using namespace std;

#define WALL_SIZE 1.0f
#define MAP_WIDTH 20
#define MAP_HEIGHT 20

/**
 * Creates a new OpenGL context and associates an OpenGL texture
 * with the output of the raycaster's main cuda program.
 */
void caster_setup(const char * window_title);

/**
 * Call after caster_setup, loads the various assets the rendering
 * engine will use to draw the scene. Including the tilesheet, skybox, and floor.
 * \param tilesheet_name Filename for the tilesheet the current map can use.
 * \param skybox_name Filename for the skybox to draw where walls/ceiling is ommited.
 * \param floor_name Filename for floor texture to draw where floor is ommitted.
 * 		This will be drawn as if it sits a few feet below the actual floor of the map.
 */
void caster_load_assets(
	const char * tilesheet_name, const char * skybox_name, const char * floor_name);

/**
 * Cleanup cuda assets when the application is ready to exit.
 */
void caster_cleanup();

/**
 * Renders a new frame of the currently loaded map. Using the input camera data.
 * \param posx X position of the camera.
 * \param posy Y position of the camera.
 * \param cam_rot Camera rotation relative to the positiev y-axis.
 */
void caster_update(float posx, float posy, float cam_rot);

#endif
