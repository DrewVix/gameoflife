#ifndef __UTIL_H__
#define __UTIL_H__

#include "cuda_runtime.h"

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) < (b) ? (b) : (a))
#endif

struct options {
	unsigned int rows;
	unsigned int cols;
	unsigned int window_width;
	unsigned int window_height;
	bool automatic;
	bool paused;
	dim3 gridSize;
	dim3 blockSize;
	int n_blocks;

	bool gui;
	bool drugs;
	unsigned int period;

	float zoom;
	float view_x;
	float view_y;
};

struct point2f {
	float x;
	float y;
};

void dumpMap(bool *, unsigned int, unsigned int);
bool is_integer(const char *);
int div_round_up(int a, int b);
bool parse_argv(struct options *, int, char **);
void random_init(bool *, unsigned int, unsigned int);
void init_cell_coords(struct point2f **coords);
void update_map(int foo);

#endif