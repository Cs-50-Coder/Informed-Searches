
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


# Pygame GUI logic
class PathfindingApp:

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent")

        self.font_sm  = pygame.font.SysFont("monospace", 12)
        self.font_md  = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_lg  = pygame.font.SysFont("monospace", 15, bold=True)

        self.num_rows = 20
        self.num_cols = 20
        self.grid = self._empty_grid()

        # cell_size is recalculated whenever the grid dimensions change
        self.cell_size = self._calc_cell_size()

        # Start and Goal are fixed until user moves them
        self.start_cell = (0, 0)
        self.goal_cell  = (self.num_rows - 1, self.num_cols - 1)

        self.algorithm      = "A*"
        self.heuristic_fn   = heuristic_manhattan
        self.heuristic_name = "Manhattan"

        self.frontier        = set()
        self.visited         = set()
        self.final_path      = []
        self.is_searching    = False
        self.search_done     = False
        self.no_path_found   = False
        self.search_gen      = None

        #  Dynamic mode 
        self.dynamic_mode = False

        # Interaction modes 
        self.placing_start = False
        self.placing_goal  = False

        # Metrics
        self.nodes_visited = 0
        self.path_length   = 0
        self.time_ms       = 0.0

        # Input boxes
        self.input_rows    = str(self.num_rows)
        self.input_cols    = str(self.num_cols)
        self.focused_input = None   

        # Timers
        self._last_search_step = 0
        self._last_dyn_spawn   = 0

        # 
        self.btn = {}  


    #HELPER FUNCTIONS

    def _empty_grid(self):
        return [[0] * self.num_cols for _ in range(self.num_rows)]

    def _calc_cell_size(self):
        size_w = GRID_AREA_W // self.num_cols
        size_h = GRID_AREA_H // self.num_rows
        return max(4, min(size_w, size_h))   # at least 4 px
    
    """Convert mouse pixel position to (row, col). Returns None if outside grid."""
    def _cell_from_mouse(self, mx, my):

        if mx < 0 or mx >= GRID_AREA_W or my < 0 or my >= GRID_AREA_H:
            return None
        col = mx // self.cell_size
        row = my // self.cell_size
        if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
            return (row, col)
        return None


    # Search Controls
    def _reset_search(self):
        self.frontier      = set()
        self.visited       = set()
        self.final_path    = []
        self.is_searching  = False
        self.search_done   = False
        self.no_path_found = False
        self.search_gen    = None
        self.nodes_visited = 0
        self.path_length   = 0
        self.time_ms       = 0.0

    def _start_search(self, from_cell=None):
        self._reset_search()
        start = from_cell if from_cell is not None else self.start_cell

        # Edge case: already at goal
        if start == self.goal_cell:
            self.final_path  = [start]
            self.path_length = 0
            self.search_done = True
            return

        # Never let dynamic obstacles block the start or goal
        sr, sc = start
        gr, gc = self.goal_cell
        self.grid[sr][sc] = 0
        self.grid[gr][gc] = 0

        args = (self.grid, start, self.goal_cell,
                self.heuristic_fn, self.num_rows, self.num_cols)

        # Pass 1: instant run just to record real timing 
        t0 = time.time()
        timing_gen = astar_search(*args) if self.algorithm == "A*" else gbfs_search(*args)
        for _, _, result in timing_gen:
            if result is not None:
                break
        self.time_ms = (time.time() - t0) * 1000

        # Pass 2: fresh generator for the animated display 
        if self.algorithm == "A*":
            self.search_gen = astar_search(*args)
        else:
            self.search_gen = gbfs_search(*args)

        self.is_searching = True

    def _do_one_search_step(self):
        if self.search_gen is None:
            return
        try:
            frontier, visited, result = next(self.search_gen)
            self.frontier      = frontier
            self.visited       = visited
            self.nodes_visited = len(visited)
            if result is not None:
                # time_ms already set from the instant pass in _start_search
                self.final_path    = result
                self.path_length   = max(len(result) - 1, 0)
                self.is_searching  = False
                self.search_done   = True
                self.no_path_found = (len(result) == 0)
                self.frontier      = set()
        except StopIteration:
            self.is_searching  = False
            self.search_done   = True
            self.no_path_found = True


    # Map Operations
    def _random_maze(self):
        self._reset_search()
        self.grid = self._empty_grid()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if (row, col) in (self.start_cell, self.goal_cell):
                    continue
                if random.random() < 0.25:
                    self.grid[row][col] = 1

    def _clear_grid(self):
        self._reset_search()
        self.grid = self._empty_grid()

    def _resize_grid(self):
        try:
            new_rows = max(5, min(40, int(self.input_rows)))
            new_cols = max(5, min(40, int(self.input_cols)))
        except ValueError:
            return

        self.num_rows = new_rows
        self.num_cols = new_cols
        self.grid     = self._empty_grid()

        self.cell_size = self._calc_cell_size()

        self.start_cell = (0, 0)
        self.goal_cell  = (self.num_rows - 1, self.num_cols - 1)

        self._reset_search()

    # Dynamic Obstacles
    def _spawn_obstacles(self):
        path_cells      = set(self.final_path)
        path_blocked    = False

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell = (row, col)
                if cell in (self.start_cell, self.goal_cell):
                    continue
                if self.grid[row][col] == 0 and random.random() < DYN_SPAWN_CHANCE:
                    self.grid[row][col] = 1
                    if cell in path_cells:
                        path_blocked = True

        if path_blocked:
            self._start_search(from_cell=self.start_cell)


    # Drawing helpers
    def _draw_btn(self, label, rect, color=COLOR_BTN):
        r = pygame.Rect(rect)
        pygame.draw.rect(self.screen, color, r, border_radius=4)
        pygame.draw.rect(self.screen, (160, 160, 160), r, 1, border_radius=4)
        txt = self.font_sm.render(label, True, COLOR_TEXT)
        self.screen.blit(txt, txt.get_rect(center=r.center))
        return r

    def _draw_input(self, key, value, label_text, x, y, w):
        lbl = self.font_sm.render(label_text, True, COLOR_TEXT)
        self.screen.blit(lbl, (x, y))
        y += lbl.get_height() + 2

        focused = (self.focused_input == key)
        box = pygame.Rect(x, y, w, 22)
        pygame.draw.rect(self.screen, (90, 90, 130) if focused else (75, 75, 75), box)
        pygame.draw.rect(self.screen, COLOR_ACCENT if focused else (110, 110, 110), box, 1)
        content = self.font_sm.render(value + ("|" if focused else ""), True, COLOR_TEXT)
        self.screen.blit(content, (box.x + 4, box.y + 3))
        return box, y + box.height + 2  # box height + 2px gap


    # Draw the grid
    def draw_grid(self):
        cs = self.cell_size

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell = (row, col)
                px = col * cs
                py = row * cs

                # Choose colour based on cell state
                if self.grid[row][col] == 1:
                    color = COLOR_WALL
                elif cell == self.start_cell:
                    color = COLOR_START
                elif cell == self.goal_cell:
                    color = COLOR_GOAL
                elif cell in self.final_path and self.search_done and not self.no_path_found:
                    color = COLOR_PATH
                elif cell in self.frontier:
                    color = COLOR_FRONTIER
                elif cell in self.visited:
                    color = COLOR_VISITED
                else:
                    color = COLOR_WHITE

                pygame.draw.rect(self.screen, color, (px, py, cs, cs))
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, (px, py, cs, cs), 1)

        for cell, letter in [(self.start_cell, "S"), (self.goal_cell, "G")]:
            row, col = cell
            cx = col * cs + cs // 2
            cy = row * cs + cs // 2
            t = self.font_md.render(letter, True, COLOR_TEXT)
            self.screen.blit(t, t.get_rect(center=(cx, cy)))

    # Draw Panel
    def draw_panel(self):
        panel_x = GRID_AREA_W   

        pygame.draw.rect(self.screen, COLOR_PANEL_BG,
                         (panel_x, 0, PANEL_W, WINDOW_H))

        x  = panel_x + 10   
        iw = PANEL_W - 20    
        y  = 10              

        btn = {}

        # Sub Helpers
        def heading(text):
            nonlocal y
            s = self.font_lg.render(text, True, COLOR_ACCENT)
            self.screen.blit(s, (x, y))
            y += s.get_height() + 3

        def small_label(text, color=COLOR_TEXT):
            nonlocal y
            s = self.font_sm.render(text, True, color)
            self.screen.blit(s, (x, y))
            y += s.get_height() + 2

        def gap(n):
            nonlocal y
            y += n

        def separator():
            nonlocal y
            pygame.draw.line(self.screen, COLOR_SEPARATOR,
                             (x, y), (x + iw, y))
            y += 5

        def full_btn(name, label, color=COLOR_BTN, h=25):
            nonlocal y
            r = self._draw_btn(label, (x, y, iw, h), color)
            btn[name] = r
            y += h + 3

        def half_btns(name_l, lbl_l, active_l, name_r, lbl_r, active_r, h=25):
            nonlocal y
            hw = (iw - 4) // 2
            col_l = COLOR_BTN_ACTIVE if active_l else COLOR_BTN
            col_r = COLOR_BTN_ACTIVE if active_r else COLOR_BTN
            btn[name_l] = self._draw_btn(lbl_l, (x,          y, hw, h), col_l)
            btn[name_r] = self._draw_btn(lbl_r, (x + hw + 4, y, hw, h), col_r)
            y += h + 3

        def input_row(key, value, label_text):
            nonlocal y
            box, y = self._draw_input(key, value, label_text, x, y, iw)
            btn["inp_" + key] = box


        heading("PATHFINDING AGENT")
        gap(2)

        # Algorithm
        small_label("Algorithm  (1=A*  2=GBFS):", COLOR_DIM_TEXT)
        half_btns("btn_astar", "A*",   self.algorithm == "A*",
                  "btn_gbfs",  "GBFS", self.algorithm == "GBFS")
        gap(2)

        # Heuristic
        small_label("Heuristic  (M=Manh  E=Eucl):", COLOR_DIM_TEXT)
        half_btns("btn_manh", "Manhattan", self.heuristic_name == "Manhattan",
                  "btn_eucl", "Euclidean", self.heuristic_name == "Euclidean")
        gap(4)
        separator()

        # Grid size
        small_label("Grid Size (5 to 40):", COLOR_DIM_TEXT)
        input_row("rows", self.input_rows, "Rows:")
        input_row("cols", self.input_cols, "Cols:")
        full_btn("btn_resize", "Resize Grid")

        # Map buttons
        full_btn("btn_random", "Random Maze  [R]", (80, 80, 30))
        full_btn("btn_clear",  "Clear Grid   [C]")
        gap(4)
        separator()

        # Move start / goal
        small_label("Move Start or Goal:", COLOR_DIM_TEXT)
        half_btns("btn_set_start", "Set Start [S]", self.placing_start,
                  "btn_set_goal",  "Set Goal  [G]", self.placing_goal)

        if self.placing_start:
            small_label("  >> Click grid to place Start", (255, 220, 80))
        elif self.placing_goal:
            small_label("  >> Click grid to place Goal",  (255, 220, 80))
        else:
            gap(14)   

        gap(2)
        separator()

        full_btn("btn_run", "RUN SEARCH  [SPACE]", COLOR_BTN_RUN, h=28)
        dyn_lbl   = "Dynamic: ON  [D]" if self.dynamic_mode else "Dynamic: OFF [D]"
        dyn_color = COLOR_BTN_DYN_ON if self.dynamic_mode else COLOR_BTN
        full_btn("btn_dynamic", dyn_lbl, dyn_color)
        gap(4)
        separator()

        # Metrics
        heading("METRICS")
        small_label(f"  Nodes Visited : {self.nodes_visited}")
        small_label(f"  Path Length   : {self.path_length}")
        small_label(f"  Search Time   : {self.time_ms:.3f} ms")
        gap(2)

        # Status
        if self.is_searching:
            small_label("  Status : Searching...", (255, 215, 0))
        elif self.no_path_found:
            small_label("  Status : No path found!", (255, 80, 80))
        elif self.search_done:
            small_label("  Status : Path found!", (80, 230, 110))
        else:
            small_label("  Status : Ready", COLOR_DIM_TEXT)

        small_label(f"  Start : {self.start_cell}", COLOR_DIM_TEXT)
        small_label(f"  Goal  : {self.goal_cell}",  COLOR_DIM_TEXT)
        gap(4)
        separator()

        # Legend
        heading("LEGEND")
        legend = [
            (COLOR_START,    "Start (S)"),
            (COLOR_GOAL,     "Goal  (G)"),
            (COLOR_WALL,     "Wall"),
            (COLOR_FRONTIER, "Frontier (yellow)"),
            (COLOR_VISITED,  "Visited  (blue)"),
            (COLOR_PATH,     "Path     (green)"),
        ]
        for color, name in legend:
            pygame.draw.rect(self.screen, color, (x, y, 12, 12))
            pygame.draw.rect(self.screen, (140, 140, 140), (x, y, 12, 12), 1)
            s = self.font_sm.render(name, True, COLOR_TEXT)
            self.screen.blit(s, (x + 16, y))
            y += 15

        # Save for click detection
        self.btn = btn

    def _handle_panel_click(self, pos):
        b = self.btn

        def hit(name):
            return name in b and b[name].collidepoint(pos)

        if hit("btn_astar"):
            self.algorithm = "A*";   self._reset_search()
        elif hit("btn_gbfs"):
            self.algorithm = "GBFS"; self._reset_search()
        elif hit("btn_manh"):
            self.heuristic_fn = heuristic_manhattan; self.heuristic_name = "Manhattan"; self._reset_search()
        elif hit("btn_eucl"):
            self.heuristic_fn = heuristic_euclidean; self.heuristic_name = "Euclidean"; self._reset_search()
        elif hit("btn_resize"):
            self._resize_grid()
        elif hit("btn_random"):
            self._random_maze()
        elif hit("btn_clear"):
            self._clear_grid()
        elif hit("btn_set_start"):
            self.placing_start = not self.placing_start
            self.placing_goal  = False
        elif hit("btn_set_goal"):
            self.placing_goal  = not self.placing_goal
            self.placing_start = False
        elif hit("btn_run"):
            self._start_search()
        elif hit("btn_dynamic"):
            self.dynamic_mode = not self.dynamic_mode
        elif hit("inp_rows"):
            self.focused_input = "rows"
        elif hit("inp_cols"):
            self.focused_input = "cols"
        else:
            self.focused_input = None
