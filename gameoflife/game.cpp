#include <stdio.h>
#include <string.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#include <GL/glut.h>

#include "util.h"
#include "gui.h"
#include "kernel.h"

bool *map;
bool *map_d;
bool *updated_map_d;
struct options opt;
FILE *mylog;
// Declaramos todas estas variables como globales para
// poder usarlas desde las funciones encargadas de mostrar
// la interfaz grafica, que no pueden recibir parámetros
// de forma configurable.

// Bucle principal en modo consola, muestra el tablero y
// llama a actualizar. Controla si estamos pausados y si
// estamos en modo automatico o manual.
void cli_main_loop()
{
	char c = 0x00;

	while (c != 'q') {
		if (c == 'p') {
			opt.paused ^= true;
		}
		else if (c == 'a') {
			opt.automatic ^= true;
		}
		// cambio de modo y pausa

		system("cls");
		dumpMap((bool *)map, opt.rows, opt.cols);
		printf("Mode: %s | Status: %s | Period: %d | Rows: %d | Cols: %d | "
			"grid.width: %d | grid.height: %d | block.width: %d | block.height: %d\n",
			(opt.automatic == true) ? "auto" : "manual",
			(opt.paused == true) ? "paused" : "running",
			opt.period,
			opt.rows,
			opt.cols,
			opt.gridSize.x,
			opt.gridSize.y,
			opt.blockSize.x,
			opt.blockSize.y);

		update_map(0);

		if (opt.automatic == true) {
			Sleep(opt.period);
		}
		else {
			c = _getch();
		}
		// en modo automatico espera <opt.period> ms, en manual
		// espera a recibir una entrada por teclado.
	}
}

// Realmente el bucle principal lo gestiona GLUT, en esta funcion
// inicializamos la GUI y lanzamos el bucle.
void gui_main_loop(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitWindowPosition(1, 1);
	glutInitWindowSize(opt.window_width, opt.window_height);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	// Con el doble buffer el cambio de un frame a otro es suave 

	glutCreateWindow("El juego de la vida");

	glClearColor(0, 0, 0, 0);
	// Cuando limpiamos el lienzo, queda color negro

	glutDisplayFunc(draw);
	//glutIdleFunc(draw);
	glutKeyboardFunc(key);
	glutSpecialFunc(specialKey);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseDrag);
	// Funciones para actualizar el lienzo y recibir
	// entrada por teclado.

	glutTimerFunc(opt.period, update_map, 0);
	// Y el primer frame se actualizara a los <opt.period>, por update_map

	opt.zoom = min(opt.window_width, opt.window_height) / min(opt.rows, opt.cols);
	if (opt.zoom == .0f) {
		opt.zoom = 1.0f;
	}
	opt.view_x = -(min(opt.window_width, opt.window_height) / (opt.zoom * opt.cols)) / 1.04f;
	opt.view_y = min(opt.window_width, opt.window_height) / (opt.zoom * opt.rows) / 1.04f;

	glutMainLoop();
}

int main(int argc, char *argv[])
{
	int device_count;
	cudaDeviceProp cuda_props;
	cudaError_t ret;
	int dim;
	int screen_width;
	int screen_height;

	if (parse_argv(&opt, argc, argv) == false){
		fprintf(stderr, "Usage: %s [-m | -a] rows cols {-nogui} {-drugs}\n", argv[0]);
		return -1;
		//char *defaults[] = { argv[0], "-a", "10000", "10000" };
		//parse_argv(&opt, 4, defaults);
	}

	cudaGetDeviceCount(&device_count);
	if (device_count < 1) {
		fprintf(stderr, "No hay ninguna GPU compatible con CUDA\n");
		return -1;
	}

	cudaGetDeviceProperties(&cuda_props, 0);
	// sharedMemPerBlock
	// maxThreadsPerBlock
	// maxThreadsPerMultiProcessor

	dim = opt.rows * opt.cols;

	screen_width = glutGet(GLUT_SCREEN_WIDTH);
	screen_height = glutGet(GLUT_SCREEN_HEIGHT);
	opt.window_width = min(screen_width, screen_height) - 80;
	opt.window_height = opt.window_width;

	map = (bool *)malloc(dim * sizeof(bool));
	random_init(map, opt.rows, opt.cols);

	ret = cudaMalloc(&map_d, dim * sizeof(bool));
	checkCudaRet(ret, "cudaMalloc", __FILE__, __LINE__);
	ret = cudaMalloc(&updated_map_d, dim * sizeof(bool));
	checkCudaRet(ret, "cudaMalloc", __FILE__, __LINE__);

	ret = cudaMemcpy(map_d, map, opt.rows * opt.cols * sizeof(bool), cudaMemcpyHostToDevice);
	checkCudaRet(ret, "cudaMemCpy", __FILE__, __LINE__);
	// Copia inicial del mapa, luego iremos cambiando map_d y updated_map_d

	opt.n_blocks = div_round_up(dim, cuda_props.maxThreadsPerBlock);

	float ratio = sqrtf((float)dim / opt.n_blocks);
	opt.gridSize.x = ceil((float)opt.rows / ratio);
	opt.gridSize.y = ceil((float)opt.cols / ratio);
	opt.blockSize.x = ceil((float)opt.cols / opt.gridSize.x);
	opt.blockSize.y = ceil((float)opt.rows / opt.gridSize.y);
	printf("%d bloques de %d x %d en grid de %d x %d\n", opt.n_blocks,
		opt.blockSize.x,
		opt.blockSize.y,
		opt.gridSize.x,
		opt.gridSize.y
		);
	// Dimensionamos el grid y el bloque segun el tamaño de la matriz:
	// Ancho * Largo = Nº de hilos = dim
	// dim / max_threads_per_block = Nº de bloques = n_blocks
	// raiz(dim / n_blocks) = ratio, el numero por el que hemos de dividir
	//						  el numero de filas y columnas para obtener
	//						  las dimensiones del grid

	opt.gridSize.z = 1;
	opt.blockSize.z = 1;

	if (opt.gui == true) {
		gui_main_loop(argc, argv);
	}
	else{
		cli_main_loop();
	}

	// Un par de figuras para pruebas
	/*
	map[1] = true;
	map[opt.cols*1 + 2] = true;
	map[opt.cols*2 + 0] = true;
	map[opt.cols*2 + 1] = true;
	map[opt.cols*2 + 2] = true;
	*/
	/*
	map[5] = true;
	map[opt.cols + 5] = true;
	map[(opt.rows - 1) * opt.cols + 5] = true;
	*/
}