#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

extern struct options opt;
extern bool *map;
extern bool *map_d;

// Volcar el estado del tablero en modo consola.
void dumpMap(bool *map, unsigned int rows, unsigned int cols)
{
	unsigned int i, j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			printf("%s ", (map[i * cols + j] == true) ? "X" : " ");
		}
		printf("\n");
	}
}

bool is_integer(const char *str)
{
	while (*str) {
		if (*str < '0' || *str > '9') {
			return false;
		}
		str++;
	}
	return true;
}

int div_round_up(int a, int b)
{
	return (a + (b - 1)) / b;
}

bool parse_argv(struct options *opt, int argc, char *argv[])
{
	int i;

	opt->gui = true;
	opt->drugs = false;
	opt->period = 40;
	opt->zoom = 1;
	opt->view_x = 0;
	opt->view_y = 0;

	if (argc < 4) {
		return false;
	}

	if (strcmp(argv[1], "-a") == 0) {
		opt->automatic = true;
	}
	else if (strcmp(argv[1], "-m") == 0) {
		opt->automatic = false;
	}
	else {
		return false;
	}

	if (!is_integer(argv[2]) || !is_integer(argv[3])) {
		return false;
	}

	for (i = 4; i < argc; i++) {
		if (strcmp(argv[i], "-nogui") == 0) {
			opt->gui = false;
		}
		else if (strcmp(argv[i], "-drugs") == 0) {
			opt->drugs = true;
		}
	}

	opt->rows = atoi(argv[2]);
	opt->cols = atoi(argv[3]);

	return true;
}

// Inicializacion aleatoria del tablero.
void random_init(bool *map, unsigned int rows, unsigned int cols) {
	unsigned int count;
	unsigned int i;
	unsigned int x, y;

	memset(map, 0, rows * cols * sizeof(bool));

	srand(time(NULL));

	count = (rows * cols) / 10 + rand() % (rows * cols);
	// Inicialmente, un numero entre 0 y filas*columnas de celdas
	// estara vivo.

	for (i = 0; i < count; i++) {
		x = rand() % cols;
		y = rand() % rows;
		// En posiciones aleatorias. Puede ser que menos de count
		// celdas resulten vivas.
		map[y * cols + x] = true;
	}
}

// Inicializa las coordenadas en pantalla para cada posicion del mapa.
void init_cell_coords(struct point2f **coords)
{
	unsigned int i, j;
	float dx, dy;
	struct point2f *lp;

	if (coords && *coords == NULL) {
		*coords = (struct point2f *) malloc(opt.rows * opt.cols * sizeof(struct point2f));
		lp = *coords;

		dx = (float)1 / min(opt.window_width, opt.window_height);
		dy = (float)1 / min(opt.window_width, opt.window_height);

		for (i = 0; i < opt.rows; i++) {
			for (j = 0; j < opt.cols; j++) {
				lp[i * opt.cols + j].x = 2 * ((float)j / min(opt.window_width, opt.window_height)) - 1 + dx;
				lp[i * opt.cols + j].y = 1 - 2 * ((float)i / min(opt.window_width, opt.window_height)) - dy;
				// Con coordenada / tamaño_coordenada obtenemos la posicion
				// relativa entre 0 y 1, pero GL mide entre -1 y 1. Por ello
				// multiplicamos por 2 y restamos 1 para X, o restamos de 1
				// para Y. Con eso obtendríamos la esquina superior izquierda.
				// Sumando dx y dy obtenemos el centro.
			}
		}
	}
}