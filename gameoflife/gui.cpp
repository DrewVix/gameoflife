#include <stdlib.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>

#include "gui.h"
#include "util.h"

extern struct options opt;
extern bool *map;
extern FILE *mylog;
// globales declaradas en game.cpp

int prev_x;
int prev_y;
// Manejo del arrastre del raton

// Actualizar el lienzo en modo grafico. Estas funciones no
// reciben nada, por eso necesitamos los datos que van a usar
// en variables globales.
void draw(void)
{
	unsigned int i, j;
	float dx, dy;
	float a, b, c;
	char status[256];
	static struct point2f *coords = NULL;
	// La primera ejecucion hara que tome valor NULL...

	init_cell_coords(&coords);
	// ... pero lo arreglamos rapido. Solo lo hara la primera vez.

	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	sprintf_s(status, 256, "Mode: %s | Status: %s | Period: %d | Rows: %d | Cols: %d | "
		"grid.width: %d | grid.height: %d | block.width: %d | block.height: %d | "
		"position.x: %f | posicion.y: %f | zoom: %f",
		(opt.automatic == true) ? "auto" : "manual",
		(opt.paused == true) ? "paused" : "running",
		opt.period,
		opt.rows,
		opt.cols,
		opt.gridSize.x,
		opt.gridSize.y,
		opt.blockSize.x,
		opt.blockSize.y,
		opt.window_width * -opt.view_x * 2,
		opt.window_height * -opt.view_y * 2,
		opt.zoom);
	glColor3f(1, 0, 0);
	renderBitmapString(-1, 0.95, 0, GLUT_BITMAP_TIMES_ROMAN_10, status);

	glScalef(opt.zoom, opt.zoom, 1);
	gluLookAt(opt.view_x, opt.view_y, 0,
		opt.view_x, opt.view_y, opt.zoom + cos(0.0f),
		0.0f, 1.0f, 0.0f);

	dx = (float)1 / min(opt.window_width, opt.window_height);
	dy = (float)1 / min(opt.window_width, opt.window_height);
	// La distancia desde el centro de la coordenada que tenemos
	// en coords hasta sus esquinas.

	glColor3f(1, 1, 1);

	glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);
	glBegin(GL_QUADS);
	// Todos los vertices que enviemos desde aqui, seran tomados como
	// vertices de un cuadrado.
	for (i = 0; i < opt.rows; i++) {
		for (j = 0; j < opt.cols; j++) {
			if (map[i * opt.cols + j] == true) {
				if (opt.drugs) {
					a = ((float)rand() / (float)(RAND_MAX));
					b = ((float)rand() / (float)(RAND_MAX));
					c = ((float)rand() / (float)(RAND_MAX));
					glColor3f(a, b, c);
				}

				glVertex3f(coords[i * opt.cols + j].x - dx, coords[i * opt.cols + j].y - dy, 0);
				glVertex3f(coords[i * opt.cols + j].x + dx, coords[i * opt.cols + j].y - dy, 0);
				glVertex3f(coords[i * opt.cols + j].x + dx, coords[i * opt.cols + j].y + dy, 0);
				glVertex3f(coords[i * opt.cols + j].x - dx, coords[i * opt.cols + j].y + dy, 0);
				// Desde el centro, los 4 vertices.
			}
		}
	}
	glEnd();

	glutSwapBuffers();
}

// Manejador de eventos de teclado en GUI.
void key(unsigned char k, int x, int y)
{
	switch (k) {
	case 'p':
		opt.paused ^= true;
		update_map(0);
		break;
	case 'q':
		exit(0);
		break;
	case 'a':
		opt.automatic ^= true;
		update_map(0);
		break;
	case 'z':
		opt.zoom += opt.zoom * 0.05f;
		break;
	case 'x':
		if (opt.zoom >= 1.05f) {
			opt.zoom -= opt.zoom * 0.05f;
		}
		break;
	case 43:	// +
		if (opt.period > 5) {
			opt.period -= 5;
		}
		break;
	case 45:	// -
		opt.period += 5;
		break;
	default:
		if (opt.automatic == false) {
			update_map(0);
		}
	}
}

void specialKey(int key, int x, int y)
{
	switch (key) {
	case GLUT_KEY_LEFT:
		opt.view_x += 0.025f;
		break;
	case GLUT_KEY_RIGHT:
		opt.view_x -= 0.025f;
		break;
	case GLUT_KEY_UP:
		opt.view_y += 0.025f;
		break;
	case GLUT_KEY_DOWN:
		opt.view_y -= 0.025f;
		break;
	}
}

// Esto es necesario para controlar correctamente el arrastre del
// raton. 
void mouse(int button, int status, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && status == GLUT_DOWN) {
		prev_x = x;
		prev_y = y;
	}
}

void mouseDrag(int x, int y)
{
	int delta_x = x - prev_x;
	int delta_y = y - prev_y;

	opt.view_x += (2 * (float)delta_x / opt.window_width) / opt.zoom;
	opt.view_y += (2 * (float)delta_y / opt.window_height) / opt.zoom;
	// con delta/dimension se consigue el porcentaje entre 0 y 1 que se
	// ha desplazado. Nuestra vista es entre -1 y 1, por lo que multiplicamos
	// por 2. Por ultimo dividimos entre el zoom para que el desplazamiento
	// sea uniforme en cualquier situacion.

	prev_x = x;
	prev_y = y;
}

// Mostrar una cadena de texto en GUI.
// Shameless copy&paste: http://www.lighthouse3d.com/tutorials/glut-tutorial/bitmap-fonts/
void renderBitmapString(
	float x,
	float y,
	float z,
	void *font,
	char *string) {

	char *c;
	glRasterPos3f(x, y, z);
	for (c = string; *c != '\0'; c++) {
		glutBitmapCharacter(font, *c);
	}
}