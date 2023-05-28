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
# 0:easy 1:normal
MODE = 1

rows = 16
cols = 17
matrix = []
FLOOR = 0
LAVA = -1
HILL = 1
WALL = 2

lava = np.empty((0, 2), dtype=int)
hill = np.empty((0, 2), dtype=int)
hill = np.empty((0, 2), dtype=int)

if MODE == 0:
    RANDOM_RATE = 0.05
    hole_on_the_wall = {1,2,5,7,9,11,13}
elif MODE == 1:
    RANDOM_RATE = 0.1
    hole_on_the_wall = {1,7,13}

for i in range(rows):
    row = []
    for j in range(cols):
        row.append(FLOOR)
    matrix.append(row)

# current_map[x][z]  -1:lava(glowstone) 0:floor  1:hill  2:wall
for i in range(rows):
    for j in range(cols):
        if random.random() < RANDOM_RATE:
            matrix[i][j] = LAVA

for i in range(rows):
    for j in range(cols):
        if random.random() < RANDOM_RATE:
            matrix[i][j] = HILL
if MODE !=0:
    for i in range(0, rows, 3):
        matrix[i][4] = LAVA
        # print(f"Change 4, {i} into Lava")

# Add wall 
for i in range(16):
    matrix[i][7] = WALL

for i in hole_on_the_wall:
    matrix[i][7]=HILL

for i in range(rows):
    print(matrix[i])

with open('matrix_data.txt', 'w') as file:
    # 遍历矩阵的每一行
    for row in matrix:
        # 将每行的元素转换为字符串，并用空格分隔
        row_str = ' '.join(map(str, row))
        # 写入txt文件
        file.write(row_str + '\n')

# 关闭txt文件
file.close()

# print("Lava:")
print("Copy below to <DrawingDecorator> in .xml")
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == LAVA:
            x = i + 1
            z = j + 1
            print(f"        <DrawBlock x=\"{x}\" y=\"45\" z=\"{z}\" type=\"glowstone\" />")
            lava = np.append(lava, [[x, z]], axis=0)

# print("Hill:")
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == HILL:
            x = i + 1
            z = j + 1
            print(f"        <DrawBlock x=\"{x}\" y=\"46\" z=\"{z}\" type=\"sandstone\" />")
            hill = np.append(hill, [[x, z]], axis=0)
print(f"        <DrawCuboid x1=\"1\"  y1=\"45\" z1=\"8\"  x2=\"16\" y2=\"47\" z2=\"8\" type=\"sandstone\" />")
for i in hole_on_the_wall:
    print(f"        <DrawBlock x=\"{i+1}\" y=\"47\" z=\"8\" type=\"air\" />")



# Store the map
if MODE == 0:
    current_map_file = 'current_map_file_easy.txt'
elif MODE ==1:
    current_map_file = 'current_map_file_normal.txt'

with open(current_map_file, 'w') as file:
    for row in matrix:
        file.write(' '.join(map(str, row)) + '\n')

# # Store the lava
# lava_file = 'lava.txt'
# with open(lava_file, 'w') as file:
#     for coordinate in lava:
#         file.write(f"       <DrawBlock x=\"{coordinate[0]}\" y=\"45\" z=\"{coordinate[1]}\" type=\"glowstone\" />\n")

# # Store the hill
# hill_file = 'hill.txt'
# with open(hill_file, 'w') as file:
#     for coordinate in hill:
#         file.write(f"       <DrawBlock x=\"{coordinate[0]}\" y=\"46\" z=\"{coordinate[1]}\" type=\"sandstone\" />\n")

# Copy2DrawingDecorator
if MODE == 0:
    copy2DrawingDecorator = 'copy2DrawingDecorator_easy.txt'
elif MODE == 1:
    copy2DrawingDecorator = 'copy2DrawingDecorator_normal.txt'
with open(copy2DrawingDecorator, 'w') as file:
    for coordinate in lava:
        file.write(f"        <DrawBlock x=\"{coordinate[0]}\" y=\"45\" z=\"{coordinate[1]}\" type=\"glowstone\" />\n")
    for coordinate in hill:
        file.write(f"        <DrawBlock x=\"{coordinate[0]}\" y=\"46\" z=\"{coordinate[1]}\" type=\"sandstone\" />\n")
    file.write(f"        <DrawCuboid x1=\"1\"  y1=\"45\" z1=\"8\"  x2=\"16\" y2=\"47\" z2=\"8\" type=\"sandstone\" />\n")
    for i in hole_on_the_wall:
        file.write(f"        <DrawBlock x=\"{i+1}\" y=\"47\" z=\"8\" type=\"air\" />\n")
