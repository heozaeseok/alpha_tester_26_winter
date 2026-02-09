import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball, Key
from minigrid.minigrid_env import MiniGridEnv
import pandas as pd
import random
import os

# --- Custom Objects ---
class HealthyTree(Box):
    def __init__(self): super().__init__(color='green')
    def can_overlap(self): return True

class BurningTree(Ball):
    def __init__(self): super().__init__(color='red')
    def can_overlap(self): return True

class BurntTree(Box):
    def __init__(self): super().__init__(color='grey')
    def can_overlap(self): return True

class ExtinguishedTree(Box):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True

class WaterTank(Key):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True

class Stone(Box):
    def __init__(self): super().__init__(color='purple') 
    def can_overlap(self): return True 

# --- Environment ---
class ForestFireEnv(MiniGridEnv):
    def __init__(self, render_mode=None, fire_spread_prob=0.01, burn_out_prob=0.001):
        # 1. CSV 데이터 로드 및 크기 계산
        self.csv_path = r"C:\Users\USER\Desktop\forest_fire\grid_analysis.csv"
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        self.max_row = int(self.df['row_index'].max()) + 1
        self.max_col = int(self.df['col_index'].max()) + 1
        
        # 가로, 세로 크기 설정 (벽 포함 +2)
        self.grid_w = self.max_col + 2
        self.grid_h = self.max_row + 2
        
        # 환경 설정 변수
        self.max_water = 2
        self.current_water = self.max_water
        self.tank_pos = (1, 1)
        self.fire_spread_prob = fire_spread_prob
        self.burn_out_prob = burn_out_prob
        
        # 보상/페널티 설정
        self.p_step = -0.01
        self.p_burnt = -1.0
        self.r_extinguish = 2.0
        
        self.fixed_tree_coords = []
        self.fixed_stone_coords = []
        self._process_csv_data()

        mission_space = MissionSpace(mission_func=lambda: "Extinguish all fires in the forest")
        
        # grid_size 대신 width, height를 명시적으로 전달 (빈 공간 제거)
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2000, 
            render_mode=render_mode,
            see_through_walls=True
        )

        # Observation: [에이전트X, 에이전트Y, 방향, 물 잔량, 가장 가까운 불X, 불Y, 불 거리, 나무 비율...]
        self.observation_space = spaces.Box(low=0, high=max(self.grid_w, self.grid_h), shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(5) # 0:좌, 1:우, 2:상, 3:하, 4:소화

    def _process_csv_data(self):
        # (row, col) -> is_tree 매핑
        grid_lookup = {(int(row['row_index']), int(row['col_index'])): int(row['is_tree']) 
                       for _, row in self.df.iterrows()}
        
        for (r, c), is_tree in grid_lookup.items():
            tx, ty = c + 1, r + 1 # MiniGrid 좌표계로 변환
            if (tx, ty) == self.tank_pos: continue

            if is_tree == 1:
                self.fixed_tree_coords.append((tx, ty))
            else:
                # 4방향 나무 둘러싸임 확인 (상하좌우가 모두 1인 빈 공간 0)
                surround = [
                    grid_lookup.get((r-1, c), 0),
                    grid_lookup.get((r+1, c), 0),
                    grid_lookup.get((r, c-1), 0),
                    grid_lookup.get((r, c+1), 0)
                ]
                if sum(surround) == 4:
                    self.fixed_stone_coords.append((tx, ty))

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(WaterTank(), *self.tank_pos)
        
        for (sx, sy) in self.fixed_stone_coords:
            self.grid.set(sx, sy, Stone())

        self.trees = []
        for (tx, ty) in self.fixed_tree_coords:
            self.grid.set(tx, ty, HealthyTree())
            self.trees.append((tx, ty))

        # 초기 화재 설정
        if len(self.trees) >= 3:
            fire_indices = random.sample(range(len(self.trees)), 3)
            for idx in fire_indices:
                fx, fy = self.trees[idx]
                self.grid.set(fx, fy, BurningTree())

        self.agent_pos = self.tank_pos
        self.agent_dir = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.current_water = self.max_water
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        reward = self.p_step
        terminated = False
        truncated = False

        # 이동 및 소화 로직
        old_pos = self.agent_pos
        if action < 4: # 이동
            if action == 0: move = np.array([-1, 0])
            elif action == 1: move = np.array([1, 0])
            elif action == 2: move = np.array([0, -1])
            elif action == 3: move = np.array([0, 1])
            
            new_pos = self.agent_pos + move
            cell = self.grid.get(*new_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = tuple(new_pos)
                if isinstance(cell, WaterTank):
                    self.current_water = self.max_water
        
        elif action == 4: # 소화
            if self.current_water > 0:
                cell = self.grid.get(*self.agent_pos)
                if isinstance(cell, BurningTree):
                    self.grid.set(*self.agent_pos, ExtinguishedTree())
                    self.current_water -= 1
                    reward += self.r_extinguish

        # 화재 확산
        self._spread_fire()

        # 종료 조건 (모든 불이 꺼지거나 나무가 다 타거나)
        fires = self._get_fire_positions()
        if len(fires) == 0:
            terminated = True
            reward += 10.0
        
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _spread_fire(self):
        fires = self._get_fire_positions()
        for fx, fy in fires:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = fx + dx, fy + dy
                if 0 < nx < self.grid_w-1 and 0 < ny < self.grid_h-1:
                    cell = self.grid.get(nx, ny)
                    if isinstance(cell, HealthyTree) and random.random() < self.fire_spread_prob:
                        self.grid.set(nx, ny, BurningTree())
            
            if random.random() < self.burn_out_prob:
                self.grid.set(fx, fy, BurntTree())

    def _get_fire_positions(self):
        fires = []
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if isinstance(self.grid.get(x, y), BurningTree):
                    fires.append((x, y))
        return fires

    def _get_obs(self):
        fires = self._get_fire_positions()
        if fires:
            dist = [np.linalg.norm(np.array(self.agent_pos) - np.array(f)) for f in fires]
            closest_fire = fires[np.argmin(dist)]
            min_dist = min(dist)
        else:
            closest_fire = (0, 0)
            min_dist = 0

        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.current_water,
            closest_fire[0], closest_fire[1], min_dist,
            len(fires), len(self.trees)
        ], dtype=np.float32)