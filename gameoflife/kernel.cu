
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdio.h>
#include <GL/glut.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"
#include "util.h"
#include "gui.h"

extern struct options opt;
extern bool *map;
extern bool *map_d;
extern bool *updated_map_d;
// Globales declaradas en game.cpp

__device__ unsigned int mod(int a, int m)
{
	int r = a % m;
	while (r < 0) {
		r += m;
	}
	return r;
}

// Actualizacion del mapa en la GPU. Tiene dos fases, separadas por syncthreads:
//	1 - Copiar en una matriz compartida por bloque las celdas correspondientes a cada
//		hilo y el "marco" que envuelve al bloque. Así evitamos leer repetidamente de
//		memoria global. 
//	2 - Cada hilo, usando la copia en memoria compartida, analiza lo que tiene alrededor
//		y guarda su estado en la matriz de resultados.
// No podemos sincronizar entre bloques, y un bloque podría pisar el mapa antes de que
// los hilos de otro pudiesen calcular su estado, por lo que usamos una segunda matriz
// para guardar el resultado.
__global__ void update_map_d(bool *map_d, bool *updated_map_d, const unsigned int rows, const unsigned int cols,
	const unsigned int block_width, const unsigned int block_height)
{
	unsigned int alive = 0;
	bool status;
	int i, j;

	const unsigned int shared_rows = block_height + 2;
	const unsigned int shared_cols = block_width + 2;
	extern __shared__ float shared_tile_ptr[];
	bool *shared_tile = (bool *)shared_tile_ptr;

	status = map_d[mod((blockIdx.y * block_height + threadIdx.y), rows) * cols + mod(blockIdx.x * block_width + threadIdx.x, cols)];
	shared_tile[(threadIdx.y + 1) * shared_cols + threadIdx.x + 1] = status;

	// Copiamos a la matriz compartida las celdas correspondientes al bloque y su entorno
	if (threadIdx.x == 0) {
		shared_tile[(threadIdx.y + 1) * shared_cols] = map_d[mod((blockIdx.y * block_height + threadIdx.y), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x - 1, cols)];
	}
	else if (threadIdx.x == block_width - 1) {
		shared_tile[(threadIdx.y + 2) * shared_cols - 1] = map_d[mod((blockIdx.y * block_height + threadIdx.y), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x + 1, cols)];
	}
	// Columna izquierda o derecha del borde del bloque?

	if (threadIdx.y == 0) {
		shared_tile[threadIdx.x + 1] = map_d[mod((blockIdx.y * block_height + threadIdx.y - 1), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x, cols)];
	}
	else if (threadIdx.y == block_height - 1) {
		shared_tile[(threadIdx.y + 2) * shared_cols + threadIdx.x + 1] = map_d[mod((blockIdx.y * block_height + threadIdx.y + 1), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x, cols)];
	}
	// Fila superior o inferior del borde del bloque?

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		shared_tile[0] = map_d[mod((blockIdx.y * block_height + threadIdx.y - 1), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x - 1, cols)];
	}
	else if (threadIdx.x == 0 && threadIdx.y == block_height - 1) {
		shared_tile[shared_cols * (shared_rows - 1)] = map_d[mod((blockIdx.y * block_height + threadIdx.y + 1), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x - 1, cols)];
	}
	else if (threadIdx.x == block_width - 1 && threadIdx.y == 0) {
		shared_tile[threadIdx.x + 2] = map_d[mod((blockIdx.y * block_height + threadIdx.y - 1), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x + 1, cols)];
	}
	else if (threadIdx.x == block_width - 1 && threadIdx.y == block_height - 1) {
		shared_tile[shared_rows * shared_cols - 1] = map_d[mod((blockIdx.y * block_height + threadIdx.y + 1), rows) * cols
			+ mod(blockIdx.x * block_width + threadIdx.x + 1, cols)];
	}
	// Alguna de las esquinas del bloque?

	__syncthreads();

	for (i = -1; i < 2; i++) {
		for (j = -1; j < 2; j++) {
			if (shared_tile[(threadIdx.y + 1 + j) * shared_cols + threadIdx.x + 1 + i] == true) {
				alive++;
			}
		}
	}

	if (status == true) {
		// Una celda viva sobrevive si no tiene 2 o 3 vivas alrededor.
		alive--;
		// Habiamos contado la propia celda
		if (alive != 2 && alive != 3) {
			status = false;
		}
	}
	else {
		// Una celda muerta revive si tiene 3 vivas alrededor
		if (alive == 3) {
			status = true;
		}
	}

	updated_map_d[mod((blockIdx.y * block_height + threadIdx.y), rows) * cols + mod(blockIdx.x * block_width + threadIdx.x, cols)] = status;
}

// Actualizar el mapa desde la CPU, llama a la GPU.
void update_map(int foo)
{
	cudaError_t ret;
	bool *tmp;

	if (opt.paused == false) {
		// En modo GUI esta funcion se llama cada cierto tiempo, por lo que hay que ver si se
		// esta pausado para no hacer nada.

		update_map_d << <opt.gridSize, opt.blockSize, (opt.blockSize.x + 2) * (opt.blockSize.y + 2) * sizeof(bool) >> >(
			map_d, updated_map_d,
			opt.rows, opt.cols,
			opt.blockSize.x, opt.blockSize.y);
		ret = cudaDeviceSynchronize();
		checkCudaRet(ret, "update_map_d launch", __FILE__, __LINE__);

		ret = cudaMemcpy(map, updated_map_d, opt.rows * opt.cols * sizeof(bool), cudaMemcpyDeviceToHost);
		checkCudaRet(ret, "cudaMemCpy", __FILE__, __LINE__);

		tmp = map_d;
		map_d = updated_map_d;
		updated_map_d = tmp;
		// Intercambiamos map_d y updated_map_d, el mapa generado en una ronda sera la base de la siguiente.

		if (opt.gui == true) {
			glutPostRedisplay();
		}
		// Tras actualizar la matriz, refrescamos el lienzo en modo grafico.

		if (opt.automatic == true) {
			glutTimerFunc(opt.period, update_map, 0);
		}
		// Si estamos en modo automatico, hacemos que se vuelva a llamar pasado el periodo
	}
}