import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball
from minigrid.minigrid_env import MiniGridEnv
import pandas as pd
import random

# --- Custom Objects ---
class HealthyTree(Box):
    def __init__(self, tree_type=1):
        # 수종에 따른 색상 매핑
        colors = {1: 'green', 2: 'yellow', 3: 'purple', 4: 'blue'}
        super().__init__(color=colors.get(tree_type, 'green'))
        self.tree_type = tree_type
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

# --- Environment ---
class ForestFireEnv(MiniGridEnv):
    def __init__(self, render_mode=None, base_fire_prob=0.01, burn_out_prob=0.05):
        # CSV 데이터 로드 및 격자 크기 설정
        self.df = pd.read_csv('SubongSan_Grid_ver2.csv')
        self.grid_w = self.df['col_index'].max() + 3 
        self.grid_h = self.df['row_index'].max() + 3
        
        # 데이터 룩업 딕셔너리 생성
        self.grid_info = {}
        for _, row in self.df.iterrows():
            self.grid_info[(int(row['col_index'])+1, int(row['row_index'])+1)] = {
                'is_tree': int(row['is_tree']),
                'elevation': float(row['elevation']),
                'slope': float(row['slope']),
                'aspect': float(row['aspect']),
                'tree_type': int(row['tree_type'])
            }

        self.base_fire_prob = base_fire_prob
        self.burn_out_prob = burn_out_prob
        
        mission_space = MissionSpace(mission_func=lambda: "Extinguish the forest fire")
        super().__init__(
            mission_space=mission_space,
            grid_size=None,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=500,
            render_mode=render_mode
        )
        self.action_space = spaces.Discrete(5) # 0:L, 1:R, 2:F, 3:Pickup, 4:Extinguish
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        tree_positions = []
        for (x, y), info in self.grid_info.items():
            if info['is_tree'] == 1:
                self.grid.set(x, y, HealthyTree(tree_type=info['tree_type']))
                tree_positions.append((x, y))
        
        if tree_positions:
            fx, fy = random.choice(tree_positions)
            self.grid.set(fx, fy, BurningTree())

        # 에이전트 초기 위치 (나무가 없는 곳)
        empty_pos = [(x, y) for x in range(1, width-1) for y in range(1, height-1) 
                     if self.grid.get(x, y) is None]
        self.agent_pos = random.choice(empty_pos) if empty_pos else (1, 1)
        self.agent_dir = 0

    def step(self, action):
        if action < 3:
            super().step(action)
        elif action == 4: # 소화 액션
            fwd_pos = self.front_pos
            cell = self.grid.get(*fwd_pos)
            if isinstance(cell, BurningTree):
                self.grid.set(*fwd_pos, ExtinguishedTree())

        self._spread_fire()
        return self._get_obs(), 0, False, self.step_count >= self.max_steps, {}

    def _spread_fire(self):
        fires = self._get_fire_positions()
        new_fires = []
        
        for fx, fy in fires:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = fx + dx, fy + dy
                cell = self.grid.get(nx, ny)
                
                if isinstance(cell, HealthyTree):
                    info_src = self.grid_info.get((fx, fy), {})
                    info_dst = self.grid_info.get((nx, ny), {})
                    p = self.base_fire_prob
                    
                    # 물리 법칙 1: 경사도(Slope) - 오르막 방향 확산 가중
                    if info_dst.get('elevation', 0) > info_src.get('elevation', 0):
                        p *= np.exp(0.069 * info_dst.get('slope', 0))
                    
                    # 물리 법칙 2: 수목 종류 - 침엽수(1.5), 활엽수(0.7)
                    t_weight = {1: 1.5, 2: 0.7, 3: 1.0, 4: 1.0}.get(info_dst.get('tree_type', 0), 1.0)
                    p *= t_weight
                    
                    # 물리 법칙 3: 사면 방향 - 남향(180도) 건조 가중치
                    aspect = info_dst.get('aspect', 0)
                    p *= (1.0 + 0.2 * np.cos(np.radians(aspect - 180)))
                    
                    if random.random() < p:
                        new_fires.append((nx, ny))
            
            if random.random() < self.burn_out_prob:
                self.grid.set(fx, fy, BurntTree())

        for nx, ny in new_fires:
            self.grid.set(nx, ny, BurningTree())

    def _get_fire_positions(self):
        return [(x, y) for x in range(self.grid_w) for y in range(self.grid_h) 
                if isinstance(self.grid.get(x, y), BurningTree)]

    def _get_obs(self):
        fires = self._get_fire_positions()
        dist_to_fire, rel_fire_pos = 1.0, np.array([0.0, 0.0])
        if fires:
            dists = [np.linalg.norm(np.array(self.agent_pos) - np.array(f)) for f in fires]
            closest_fire = fires[np.argmin(dists)]
            dist_to_fire = min(dists) / max(self.grid_w, self.grid_h)
            rel_fire_pos = (np.array(closest_fire) - np.array(self.agent_pos)) / max(self.grid_w, self.grid_h)
        
        return np.array([self.agent_pos[0]/self.grid_w, self.agent_pos[1]/self.grid_h, 
                         rel_fire_pos[0], rel_fire_pos[1], dist_to_fire], dtype=np.float32)