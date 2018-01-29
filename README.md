# raycaster-cuda

## Overview

Implements a [raycaster](https://en.wikipedia.org/wiki/Ray_casting) in CUDA. Rendered in real time to an OpenGL texture buffer. Currently only tested to work on Linux based systems, but could easily be ported to others with modifications to the README. Should transition the project to CMAKE or other build system in the future.

![alt tag](https://raw.githubusercontent.com/imalerich/raycaster-cuda/master/img/scrot.png)

## Dependencies

- [cuda](https://www.nvidia.com/object/cuda_home_new.html)
- [glfw](www.glfw.org)
- [glew](http://glew.sourceforge.net)
