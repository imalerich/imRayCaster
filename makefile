EXE = raycaster-cuda
OBJ = main.o file_io.o gl_util.o raycaster.o
LIB = -lm -lglfw -lGLEW -lGL -lGLU -lX11 -lXxf86vm -lXrandr -lXi -lglut -lcuda

all: $(OBJ)
	nvcc -o $(EXE) $(OBJ) $(LIB)

main.o: src/main.cu gl_util.o raycaster.o
	nvcc -c src/main.cu -Istb

raycaster.o: src/raycaster.cu src/raycaster.cu
	nvcc -c src/raycaster.cu -Istb

gl_util.o: src/gl_util.cpp src/gl_util.h file_io.o
	nvcc -c src/gl_util.cpp

file_io.o: src/file_io.cpp src/file_io.h
	nvcc -c src/file_io.cpp

clean:
	rm -rf $(EXE)
	rm -rf *.o
