import random
import numpy as np
import os
import pickle
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
# 0:easy 1:normal 2:else
MODE = 2

# 0:create new map 1:read map 2:test
CREATE = 0
print("Create a new map or Read a map file")
print("Create : 0")
print("Read   : 1")
CREATE = input("CREATE=")
CREATE = int(CREATE)

print("The MODE of the map is:")
print("0:easy  1:normal  2:else")
MODE = input("MODE=")
MODE = int(MODE)

rows = 16
cols = 17
matrix = []
FLOOR = 0
LAVA = -1
HILL = 1
WALL = 2
hole_on_the_wall = {}
current_map_file = ""
EDGE = "(20,-9999)"

lava = np.empty((0, 2), dtype=int)
hill = np.empty((0, 2), dtype=int)
diamond_block = np.empty((0, 2), dtype=int)

if MODE == 0 :
    # easy
    RANDOM_RATE = 0.05
    # hole_on_the_wall = {1,2,5,7,9,11,13}
    hole_on_the_wall = {1,2,4,5,7,9,11,13,14}
elif MODE == 1:
    # normal
    RANDOM_RATE = 0.1
    # hole_on_the_wall = {1,7,13}
    hole_on_the_wall = {1,6,7,12,13}
elif MODE == 2:
    difficulty = input("difficulty=")
    # RANDOM_RATE = input("RANDOM_RATE=")
    # print("Hole on the wall:")
    # print("1 : {1,2,5,7,9,11,13}")
    # print("2 : {1,5,7,11,13}")
    # print("3 : {1,7,13}")
    # SELECTION = input("Selection=")
    # if SELECTION == 1:
    #     hole_on_the_wall = {1,2,5,7,9,11,13}
    # elif SELECTION == 2:
    #     hole_on_the_wall = {1,5,7,11,13}
    # elif SELECTION == 3:
    #     hole_on_the_wall = {1,7,13}


if CREATE == 0 :
    # create

    # initialize the matrix
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(FLOOR)
        matrix.append(row)

    if MODE == 2:
        current_map_file = "current_map_file_"+ difficulty + ".txt"
        print("RANDOM_RATE of easy is 0.05 , and 0.1 for normal")
        RANDOM_RATE = input("RANDOM_RATE=")
        RANDOM_RATE = float(RANDOM_RATE)
        print("Hole on the wall:")
        print("1 : {1,2,4,5,7,8,9,12,13,14}")
        print("2 : {1,2,4,5,7,9,11,13,14}")
        print("3 : {1,6,7,12,13}")
        SELECTION = input("Selection=")
        SELECTION = int(SELECTION)
        if SELECTION == 1:
            hole_on_the_wall = {1,2,4,5,7,8,9,12,13,14}
        elif SELECTION == 2:
            hole_on_the_wall = {1,2,4,5,7,9,11,13,14}
        elif SELECTION == 3:
            hole_on_the_wall = {1,6,7,12,13}
    elif MODE == 0:
        current_map_file = 'current_map_file_easy.txt'
    elif MODE == 1:
        current_map_file = 'current_map_file_normal.txt'
    # current_map[x][z]  -1:lava(glowstone) 0:floor  1:hill  2:wall
    # Add lava and hill into the map
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

    # Add wall into the map
    for i in range(16):
        matrix[i][7] = WALL

    # print(f"hole_on_the_wall = {hole_on_the_wall}")
    for i in hole_on_the_wall:
        matrix[i][7]=HILL

    for i in range(rows):
        print(matrix[i])

elif CREATE == 1:
    # read
    if MODE == 0:
        current_map_file = 'current_map_file_easy.txt'
    elif MODE == 1:
        current_map_file = 'current_map_file_normal.txt'
    elif MODE == 2:
        current_map_file = "current_map_file_"+ difficulty + ".txt"

    # with open(current_map_file, 'r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         row = [int(num) for num in line.strip().split()]
    #         matrix.append(row)

    with open(current_map_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:-1]:
            row = line.strip().split(' ')
            tuple_row = [eval(item) for item in row]
            temp = []
            for data in tuple_row[1:-1]:
                temp.append(data[0])
            matrix.append(temp)



print("Copy below to <DrawingDecorator> in .xml")
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == LAVA:
            x = i + 1
            z = j + 1
            print(f"        <DrawBlock x=\"{x}\" y=\"45\" z=\"{z}\" type=\"glowstone\" />")
            lava = np.append(lava, [[x, z]], axis=0)

for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == HILL:
            x = i + 1
            z = j + 1
            print(f"        <DrawBlock x=\"{x}\" y=\"46\" z=\"{z}\" type=\"sandstone\" />")
            hill = np.append(hill, [[x, z]], axis=0)
# Add wall
print(f"        <DrawCuboid x1=\"1\"  y1=\"45\" z1=\"8\"  x2=\"16\" y2=\"47\" z2=\"8\" type=\"sandstone\" />")
# Add hole on the wall
for i in hole_on_the_wall:
    print(f"        <DrawBlock x=\"{i+1}\" y=\"47\" z=\"8\" type=\"air\" />")
# Add diamond_block
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == FLOOR:
            if j == 8 or j == 9:
                x = i + 1
                z = j + 1
                print(f"        <DrawBlock x=\"{x}\" y=\"45\" z=\"{z}\" type=\"diamond_block\" />")
                diamond_block = np.append(hill, [[x, z]], axis=0)





# Store the map
# if CREATE == 2:
#     current_map_file = 'current_map_file_test.txt'
# else:
#     if MODE == 0:
#         current_map_file = 'current_map_file_easy.txt'
#     elif MODE ==1:
#         current_map_file = 'current_map_file_normal.txt'

# # only store the height
# if CREATE == 0:
#     with open(current_map_file, 'w') as file:
#         for row in matrix:
#             file.write(' '.join(map(str, row)) + '\n')

# Copy2DrawingDecorator
if MODE == 0:
    copy2DrawingDecorator = 'copy2DrawingDecorator_easy.txt'
elif MODE == 1:
    copy2DrawingDecorator = 'copy2DrawingDecorator_normal.txt'
elif MODE == 2:
    copy2DrawingDecorator = "copy2DrawingDecorator_" + difficulty + ".txt"
with open(copy2DrawingDecorator, 'w') as file:
    for coordinate in lava:
        file.write(f"        <DrawBlock x=\"{coordinate[0]}\" y=\"45\" z=\"{coordinate[1]}\" type=\"glowstone\" />\n")
    for coordinate in hill:
        file.write(f"        <DrawBlock x=\"{coordinate[0]}\" y=\"46\" z=\"{coordinate[1]}\" type=\"sandstone\" />\n")
    file.write(f"        <DrawCuboid x1=\"1\"  y1=\"45\" z1=\"8\"  x2=\"16\" y2=\"47\" z2=\"8\" type=\"sandstone\" />\n")
    for i in hole_on_the_wall:
        file.write(f"        <DrawBlock x=\"{i+1}\" y=\"47\" z=\"8\" type=\"air\" />\n")
    for coordinate in diamond_block:
        file.write(f"        <DrawBlock x=\"{coordinate[0]}\" y=\"45\" z=\"{coordinate[1]}\" type=\"diamond_block\" />\n")

# ADD MORE INFORMATION TO MATRIX
new_matrix = []
temp = []
# for i in range(rows+2):
#     temp.append(EDGE)
# new_matrix.append(temp)
temp = [EDGE] * (rows+2)
new_matrix.append(temp)

for row in matrix:
    temp2 = []
    temp2.append(EDGE)
    for i in range(len(row)):
        if row[i] == -1:
            row[i] = "(-1,-1)"
        else:
            if i == 8 or i == 9:
                row[i] = "(" + str(row[i]) + ",1)"
            else:
                row[i] = "(" + str(row[i]) + ",0)"
        temp2.append(row[i])
    temp2.append(EDGE)
    new_matrix.append(temp2)
temp = [EDGE] * (rows+2)
new_matrix.append(temp)

# Store the map information
if CREATE == 0:
    with open(current_map_file, 'w') as file:
        for row in new_matrix:
            file.write(' '.join(row) + '\n')

# Code to read map data
# matrix = []
# with open('current_map_file_999.txt', 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         row = line.strip().split(' ')
#         tuple_row = [eval(item) for item in row]
#         temp = []
#         for data in tuple_row:
#             temp.append(data[0])
#         matrix.append(temp)
# print(matrix) 