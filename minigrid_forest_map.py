import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball, Key
from minigrid.minigrid_env import MiniGridEnv
import pandas as pd
import numpy as np
import time
import random
import os

# --- Custom Objects ---
class HealthyTree(Box):
    def __init__(self): super().__init__(color='green')
    def can_overlap(self): return True 

class BurningTree(Ball):
    def __init__(self): super().__init__(color='red')
    def can_overlap(self): return True

class WaterTank(Key):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True 

class Stone(Box):
    def __init__(self): 
        super().__init__(color='purple')
    def can_overlap(self): 
        return True 

# --- Visualization Environment ---
class ForestFireMapViewer(MiniGridEnv):
    def __init__(self, render_mode="human"):
        self.csv_path = r"C:\Users\USER\Desktop\forest_fire\grid_analysis.csv"
        
        if not os.path.exists(self.csv_path):
            print(f"Error: 파일을 찾을 수 없습니다: {self.csv_path}")
            return

        self.df = pd.read_csv(self.csv_path)
        self.max_row = int(self.df['row_index'].max()) + 1
        self.max_col = int(self.df['col_index'].max()) + 1
        
        self.grid_w = self.max_col + 2
        self.grid_h = self.max_row + 2

        self.tank_pos = (1, 1)
        self.fixed_tree_coords = []
        self.fixed_stone_coords = []
        
        self._load_from_csv()

        mission_space = MissionSpace(mission_func=lambda: "Map Visualization Mode")
        
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=100,
            render_mode=render_mode,
            see_through_walls=True
        )
        self.highlight = False 

    def _load_from_csv(self):
        grid_data = {(int(row['row_index']), int(row['col_index'])): int(row['is_tree']) 
                     for _, row in self.df.iterrows()}
        
        for (r, c), is_tree in grid_data.items():
            tx, ty = c + 1, r + 1
            if (tx, ty) == self.tank_pos: continue

            if is_tree == 1:
                self.fixed_tree_coords.append((tx, ty))
            else:
                surround = [
                    grid_data.get((r-1, c), 0),
                    grid_data.get((r+1, c), 0),
                    grid_data.get((r, c-1), 0),
                    grid_data.get((r, c+1), 0)
                ]
                if sum(surround) == 4:
                    self.fixed_stone_coords.append((tx, ty))

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(WaterTank(), *self.tank_pos)
        
        for (sx, sy) in self.fixed_stone_coords:
            if 0 < sx < width and 0 < sy < height:
                self.grid.set(sx, sy, Stone())

        self.trees = []
        for (tx, ty) in self.fixed_tree_coords:
            if 0 < tx < width and 0 < ty < height:
                self.grid.set(tx, ty, HealthyTree())
                self.trees.append((tx, ty))

        if hasattr(self, 'trees') and len(self.trees) >= 5:
            fire_indices = random.sample(range(len(self.trees)), 5)
            for idx in fire_indices:
                fx, fy = self.trees[idx]
                self.grid.set(fx, fy, BurningTree())

        self.agent_pos = self.tank_pos
        self.agent_dir = 0

if __name__ == "__main__":
    env = ForestFireMapViewer(render_mode="human")
    if hasattr(env, 'width'):
        env.reset()
        env.render()
        print(f"Map Rendered: {env.width}x{env.height}")
        time.sleep(10)
        env.close()