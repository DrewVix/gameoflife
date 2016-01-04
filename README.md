# gameoflife
CUDA implementation of Game of Life

This was an assignment for the sobject "Ampliación de programación Avanzada" at UAH, coded by @ca0s and @Tokoro360

## Dependencies
- CUDA toolkit > 5.5 (
- https://developer.nvidia.com/cg-toolkit-download (GLUT)

## Usage
$ ./gameoflife [-a/-m] rows cols [-nogui] [-drugs]
Where:
- -a / -m stands for automatic / manual mode. Automatic generates a round automatically, manual waits for the user to press any key
- rows / cols determine the size of the grid
- -nogui makes the program use the console to print the grid
- -drugs will make your screen trippy

## GUI controls
- "z" / "x" to zoom / unzoom
- arrow keys / mouse drag&drop for movement
- "p" to pause / unpause
- "a" to automatic mode on / off
- "q" to quit
- any other key to generate the next round when automatic mode is off

## Screenshots
![small grid img](gameoflife/screenshots/gameoflife.PNG?raw=true "Small grid")
![small grid img](gameoflife/screenshots/gameoflife2.PNG?raw=true "Small grid")
![big grid img](gameoflife/screenshots/gameoflife4.PNG?raw=true "Big grid")
![bigger grid img](gameoflife/screenshots/gameoflife3.PNG?raw=true "Bigger grid")
