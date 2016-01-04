#ifndef __GUI_H__
#define __GUI_H__

void draw(void);
void key(unsigned char k, int x, int y);
void specialKey(int key, int x, int y);
void mouse(int button, int status, int x, int y);
void mouseDrag(int x, int y);
void mouseButton(int button, int state, int x, int y);

void renderBitmapString(float x, float y, float z, void *font, char *string);

#endif