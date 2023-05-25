import random
import numpy as np
import os
'''
This program generates a random map file and stores its content in the files "current_map_file.txt,"
"lava.txt," and "hill.txt." Copy the text from "lava.txt" and "hill.txt" or the output in terminal into the <DrawingDecorator> 
section of the .xml file to create a map.

In the "current_map_file.txt" file, the following mappings are used: 1 represents a floor, 
2 represents lava, and 3 represents a hill.

You can determine the surrounding blocks based on the current x and z coordinates. 
Since the map ranges from (1,1) to (16,17), please use "current_map[x-1][z-1]" to access the coordinates.
After that, you can use x+/-1 and z+/-1 to get the blocks around your coordinate.
'''
rows = 16
cols = 17
matrix = []
lava = np.empty((0, 2), dtype=int)
hill = np.empty((0, 2), dtype=int)

for i in range(rows):
    row = []
    for j in range(cols):
        row.append(1)
    matrix.append(row)

# current_map[x][z]  1:floor  2:lava  3:hill
for i in range(rows):
    for j in range(cols):
        if random.random() < 0.1:
            matrix[i][j] = 2

for i in range(rows):
    for j in range(cols):
        if random.random() < 0.1:
            matrix[i][j] = 3

for i in range(0, rows, 3):
    matrix[i][4] = 2
    # print(f"Change 4, {i} into Lava")

for i in range(rows):
    print(matrix[i])

# print("Lava:")
print("Copy below to <DrawingDecorator> in .xml")
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == 2:
            x = i + 1
            z = j + 1
            print(f"        <DrawBlock x=\"{x}\" y=\"45\" z=\"{z}\" type=\"glowstone\" />")
            lava = np.append(lava, [[x, z]], axis=0)

# print("Hill:")
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == 3:
            x = i + 1
            z = j + 1
            print(f"        <DrawBlock x=\"{x}\" y=\"46\" z=\"{z}\" type=\"sandstone\" />")
            hill = np.append(hill, [[x, z]], axis=0)

# Store the map
current_map_file = 'current_map_file.txt'
with open(current_map_file, 'w') as file:
    for row in matrix:
        file.write(' '.join(map(str, row)) + '\n')

# Store the lava
lava_file = 'lava.txt'
with open(lava_file, 'w') as file:
    for coordinate in lava:
        file.write(f"       <DrawBlock x=\"{coordinate[0]}\" y=\"45\" z=\"{coordinate[1]}\" type=\"glowstone\" />\n")

# Store the hill
hill_file = 'hill.txt'
with open(hill_file, 'w') as file:
    for coordinate in hill:
        file.write(f"       <DrawBlock x=\"{coordinate[0]}\" y=\"46\" z=\"{coordinate[1]}\" type=\"sandstone\" />\n")
