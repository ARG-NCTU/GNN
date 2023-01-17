import numpy as np
import math
import csv

const = 100
MAP_WIDTH = const
MAP_HEIGHT = const

map_arr = []

with open('map.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    element = 0
    for i in range(MAP_HEIGHT):
        row_arr = []
        for j in range(MAP_WIDTH):
            if i == 0 or i == MAP_HEIGHT - 1:
            # if i == 0:
                element = 1
            elif j == 0 or j == MAP_WIDTH - 1:
            # elif j == 0:
                element = 1
            else:
                element = 0
            row_arr.append(element)
        writer.writerow(row_arr)