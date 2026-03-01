
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