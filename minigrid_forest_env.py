import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, Ball, Box
from minigrid.minigrid_env import MiniGridEnv
import pandas as pd
import random

class HealthyTree(Box):
    def __init__(self, tree_type=1):
        # [수정] 4번 타입의 색상을 grey에서 purple로 변경
        colors = {1: 'red', 2: 'green', 3: 'yellow', 4: 'purple'}
        super().__init__(color=colors.get(tree_type, 'green'))
        self.tree_type = tree_type
    def can_overlap(self): return True

class BurningTree(Ball):
    def __init__(self, prob=0.01): 
        super().__init__(color='red')
        self.spread_prob = prob
    def can_overlap(self): return True

# [추가] 화재가 완전히 소실된 나무 클래스 (회색)
class BurntTree(Box):
    def __init__(self):
        super().__init__(color='grey')
    def can_overlap(self): return True

class ForestFireEnv(MiniGridEnv):
    def __init__(self, render_mode="human"):
        # --- [환경 설정 모수 (Hyperparameters)] ---
        self.grid_w, self.grid_h = 50, 50
        self.shift_y = 4            
        self.base_prob = 0.005      # 기본 확산 확률
        self.base_burnout_prob = 0.0005 # 기본 소실 확률
        self.ammo_limit = 3         
        self.max_steps = 1000       
        self.depot_pos = (25, 18)   
        
        self.tree_weights = {0: 0.0, 1: 1.5, 2: 0.8, 3: 1.2, 4: 1.0}
        self.slope_norm = 45.0      
        self.wind_impact = 3.0      
        # ------------------------------------------

        self.df = pd.read_csv(r"C:\Users\USER\Desktop\forest_fire\subongsan_integrated_final.csv")
        self.terrain_data = {}
        self.initial_tree_count = 0
        
        for _, row in self.df.iterrows():
            tx = int(row['col_index']) + 1
            ty = int(row['row_index']) + 1 + self.shift_y 
            
            if 0 < tx <= self.grid_w and 0 < ty <= self.grid_h:
                self.terrain_data[(tx, ty)] = {
                    'slope': row['slope'],
                    'aspect': row['aspect'],
                    'tree_type': int(row['tree_type']),
                    'is_tree': int(row['is_tree'])
                }
                if row['is_tree'] == 1: self.initial_tree_count += 1

        mission_space = MissionSpace(mission_func=lambda: "extinguish all fires")
        super().__init__(
            mission_space=mission_space, width=self.grid_w+2, height=self.grid_h+2,
            max_steps=self.max_steps, render_mode=render_mode, agent_view_size=201
        )
        
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        self.grid.set(self.depot_pos[0], self.depot_pos[1], Floor(color='blue'))
        
        self.fire_coords = set()
        self.ammo = self.ammo_limit
        
        tree_positions = []
        for (tx, ty), data in self.terrain_data.items():
            if data['is_tree'] == 1:
                self.grid.set(tx, ty, HealthyTree(tree_type=data['tree_type']))
                tree_positions.append((tx, ty))

        self.agent_pos = self.depot_pos
        self.agent_dir = 0
        
        self.wind_mode = random.choice(['N', 'S', 'E', 'W'])
        wind_obs_map = {'N': (0,0), 'S': (1,0), 'E': (0,1), 'W': (1,1)}
        self.wind_obs = wind_obs_map[self.wind_mode]
        
        if len(tree_positions) >= 2:
            for fx, fy in random.sample(tree_positions, 1):
                p = self.calculate_spread_prob(fx, fy)
                self.grid.set(fx, fy, BurningTree(prob=p))
                self.fire_coords.add((fx, fy))

    def calculate_spread_prob(self, x, y):
        data = self.terrain_data.get((x, y), None)
        if not data or data['is_tree'] == 0: return 0.0
        
        tw = self.tree_weights.get(data['tree_type'], 1.0)
        sw = 1.0 + (data['slope'] / self.slope_norm)
        
        if data['slope'] == 0 or data['aspect'] == -1:
            aw = 1.0
        else:
            wind_deg = {'N': 0, 'S': 180, 'E': 90, 'W': 270}[self.wind_mode]
            diff = abs(data['aspect'] - wind_deg)
            angle_diff = min(diff, 360 - diff)
            aw = 2.0 - 1.5 * (angle_diff / 180.0) 
        
        return self.base_prob * tw * sw * aw

    def calculate_burnout_prob(self, x, y):
        data = self.terrain_data.get((x, y), None)
        if not data or data['is_tree'] == 0: return 0.0
        
        tw = self.tree_weights.get(data['tree_type'], 1.0)
        sw = 1.0 + (data['slope'] / self.slope_norm)
        
        if data['slope'] == 0 or data['aspect'] == -1:
            aw = 1.0
        else:
            wind_deg = {'N': 0, 'S': 180, 'E': 90, 'W': 270}[self.wind_mode]
            diff = abs(data['aspect'] - wind_deg)
            angle_diff = min(diff, 360 - diff)
            aw = 2.0 - 1.5 * (angle_diff / 180.0) 
        
        return self.base_burnout_prob * tw * sw * aw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_obs(), {}
    
    def _get_surrounding_healthy_trees(self, x, y):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 < nx <= self.grid_w and 0 < ny <= self.grid_h:
                    if isinstance(self.grid.get(nx, ny), HealthyTree):
                        count += 1
        return count

    def _get_danger_score(self, fx, fy):
        obj = self.grid.get(fx, fy)
        if isinstance(obj, BurningTree):
            return obj.spread_prob * self._get_surrounding_healthy_trees(fx, fy)
        return 0.0

    def step(self, action):
        self.step_count += 1
        reward = -0.1 
        
        d_close_pre, d_danger_pre = self._get_fire_distances()

        moves = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}
        dx, dy = moves[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        if 0 < nx <= self.grid_w and 0 < ny <= self.grid_h:
            self.agent_pos = (nx, ny)

        if self.agent_pos == self.depot_pos:
            self.ammo = self.ammo_limit

        d_close_post, d_danger_post = self._get_fire_distances()
        if d_close_post < d_close_pre: reward += 0.01
        if d_danger_post < d_danger_pre: reward += 0.01

        curr_obj = self.grid.get(*self.agent_pos)
        if isinstance(curr_obj, BurningTree):
            danger_score = self._get_danger_score(self.agent_pos[0], self.agent_pos[1])
            reward += 5.0 + (danger_score * 10.0)
            
            self.grid.set(*self.agent_pos, Floor())
            self.fire_coords.discard(self.agent_pos)
            self.ammo -= 1
            
            if self.ammo <= 0:
                self.agent_pos = self.depot_pos
                self.ammo = self.ammo_limit

        spread_count = self._spread_fire()
        reward -= (spread_count * 1.0)

        terminated = len(self.fire_coords) == 0
        truncated = self.step_count >= self.max_steps
        
        current_trees = sum(1 for (tx, ty) in self.terrain_data if isinstance(self.grid.get(tx, ty), HealthyTree))
        
        if current_trees < (self.initial_tree_count * 0.5):
            terminated = True
            reward -= 100.0
        
        if terminated and len(self.fire_coords) == 0:
            reward += current_trees * 10.0

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_fire_distances(self):
        if not self.fire_coords: return 0, 0
        ax, ay = self.agent_pos
        dists = [np.sqrt((fx-ax)**2 + (fy-ay)**2) for fx, fy in self.fire_coords]
        
        danger_list = [(self._get_danger_score(fx, fy), fx, fy) for fx, fy in self.fire_coords]
        _, dfx, dfy = max(danger_list)
        dist_danger = np.sqrt((dfx-ax)**2 + (dfy-ay)**2)
        
        return min(dists), dist_danger

    def _get_obs(self):
        ax, ay = self.agent_pos
        if not self.fire_coords: return np.zeros(9, dtype=np.float32)

        dists = [((fx-ax)**2 + (fy-ay)**2, fx, fy) for fx, fy in self.fire_coords]
        _, cfx, cfy = min(dists)
        
        danger_list = [(self._get_danger_score(fx, fy), fx, fy) for fx, fy in self.fire_coords]
        _, dfx, dfy = max(danger_list)

        return np.array([
            ax, ay, 
            self.wind_obs[0], self.wind_obs[1],
            self.ammo,
            cfx - ax, cfy - ay,
            dfx - ax, dfy - ay
        ], dtype=np.float32)
    
    def _spread_fire(self):
        new_fires = set()
        burnt_out_fires = set() # [추가] 소실된 화재 좌표 저장
        
        for fx, fy in list(self.fire_coords):
            # 1. 소실 체크 (일정 확률로 스스로 꺼지고 회색이 됨)
            burnout_prob = self.calculate_burnout_prob(fx, fy)
            if random.random() < burnout_prob:
                burnt_out_fires.add((fx, fy))
                continue # 소실된 셀은 주변으로 불을 퍼뜨리지 않음
                
            # 2. 불 확산 체크
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = fx + dx, fy + dy
                if (nx, ny) in self.terrain_data and isinstance(self.grid.get(nx, ny), HealthyTree):
                    prob = self.calculate_spread_prob(nx, ny)
                    if random.random() < prob:
                        new_fires.add((nx, ny))
        
        # [수정] 소실된 화재 처리 (회색으로 덮고 화재 좌표에서 제거)
        for bx, by in burnt_out_fires:
            self.grid.set(bx, by, BurntTree())
            self.fire_coords.discard((bx, by))
        
        # 새로운 화재 처리
        for nx, ny in new_fires:
            p = self.calculate_spread_prob(nx, ny)
            self.grid.set(nx, ny, BurningTree(prob=p))
            self.fire_coords.add((nx, ny))
            
        return len(new_fires)