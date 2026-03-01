
import pygame
import sys
import math
import heapq
import random
import time

#Window Dimensions
WINDOW_W = 1100   
WINDOW_H = 750    
PANEL_W  = 290    

#Map Grid Dimensions
GRID_AREA_W = WINDOW_W - PANEL_W  
GRID_AREA_H = WINDOW_H            

FPS = 60

SEARCH_STEP_MS = 20

#Dynamic walls logic
DYN_SPAWN_INTERVAL_MS = 3000   
DYN_SPAWN_CHANCE      = 0.001  

COLOR_WHITE      = (255, 255, 255)
COLOR_WALL       = (35,  35,  35 )
COLOR_START      = (0,   200, 80 )   # green
COLOR_GOAL       = (210, 50,  50 )   # red
COLOR_FRONTIER   = (255, 215, 0  )   # yellow
COLOR_VISITED    = (80,  130, 230)   # blue
COLOR_PATH       = (0,   220, 90 )   # bright green
COLOR_GRID_LINE  = (160, 160, 160)
COLOR_BG         = (200, 200, 200)
COLOR_PANEL_BG   = (45,  45,  45 )
COLOR_TEXT       = (240, 240, 240)
COLOR_DIM_TEXT   = (150, 150, 150)
COLOR_ACCENT     = (90,  170, 255)
COLOR_BTN        = (70,  70,  110)
COLOR_BTN_ACTIVE = (50,  170, 50 )
COLOR_BTN_RUN    = (30,  140, 30 )
COLOR_BTN_DYN_ON = (170, 50,  50 )
COLOR_SEPARATOR  = (90,  90,  90 )

#Heuristic calculations
def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic_euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Grid Helpers
def get_neighbors(cell, num_rows, num_cols, grid):
    row, col = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < num_rows and 0 <= c < num_cols and grid[r][c] == 0:
            yield (r, c)

def rebuild_path(came_from, end_cell):
    path = []
    cell = end_cell
    while cell in came_from:
        path.append(cell)
        cell = came_from[cell]
    path.append(cell)
    path.reverse()
    return path


"""
    A* Search  ->  f(n) = g(n) + h(n)
    g(n) = pichlay path ki cost
    h(n) = current node ki heuristics

    Guaranteed to find the SHORTEST path.
"""
def astar_search(grid, start, goal, heuristic, num_rows, num_cols):

    h_start = heuristic(start, goal)

    heap = [(h_start, h_start, 0, start)]

    came_from = {}
    g_score   = {start: 0}
    visited   = set()
    in_queue  = {start}

    while heap:
        f, h, g, current = heapq.heappop(heap)

        if current in visited:
            continue

        in_queue.discard(current)
        visited.add(current)

        if current == goal:
            yield in_queue.copy(), visited.copy(), rebuild_path(came_from, goal)
            return

        for nb in get_neighbors(current, num_rows, num_cols, grid):
            new_g = g + 1
            if nb not in g_score or new_g < g_score[nb]:
                g_score[nb] = new_g
                came_from[nb] = current
                new_h = heuristic(nb, goal)
                new_f = new_g + new_h
                heapq.heappush(heap, (new_f, new_h, new_g, nb))
                in_queue.add(nb)

        yield in_queue.copy(), visited.copy(), None

    yield set(), visited.copy(), []   # no path

"""
    Greedy Best-First Search  ->  f(n) = h(n) only
    Rushes toward the goal, ignores actual path cost.
    Faster but NOT guaranteed to find the shortest path.
"""
def gbfs_search(grid, start, goal, heuristic, num_rows, num_cols):

    counter = 0
    heap = [(heuristic(start, goal), counter, start)]

    came_from = {}
    visited   = set()
    in_queue  = {start}

    while heap:
        _, _, current = heapq.heappop(heap)

        if current in visited:
            continue

        in_queue.discard(current)
        visited.add(current)

        if current == goal:
            yield in_queue.copy(), visited.copy(), rebuild_path(came_from, goal)
            return

        for nb in get_neighbors(current, num_rows, num_cols, grid):
            if nb not in visited and nb not in in_queue:
                came_from[nb] = current
                counter += 1
                heapq.heappush(heap, (heuristic(nb, goal), counter, nb))
                in_queue.add(nb)

        yield in_queue.copy(), visited.copy(), None

    yield set(), visited.copy(), []

    