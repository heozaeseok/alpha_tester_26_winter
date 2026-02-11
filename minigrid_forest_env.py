import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball
from minigrid.minigrid_env import MiniGridEnv
import pandas as pd
import random

# --- 1. 객체 정의 ---
class HealthyTree(Box):
    def __init__(self, tree_type=1):
        colors = {1: 'red', 2: 'green', 3: 'yellow', 4: 'grey'}
        super().__init__(color=colors.get(tree_type, 'green'))
        self.tree_type = tree_type
    def can_overlap(self): return True

class BurningTree(Ball):
    def __init__(self, spread_prob=0.01): 
        super().__init__(color='red')
        self.spread_prob = spread_prob
    def can_overlap(self): return True

class ExtinguishedTree(Box):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True

class BurntTree(Box):
    def __init__(self): super().__init__(color='grey')
    def can_overlap(self): return True

class ForestFireEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.csv_path = r"C:\Users\USER\Desktop\forest_fire\subongsan_integrated_final.csv"
        self.df = pd.read_csv(self.csv_path)
        
        self.grid_w = self.df['col_index'].max() + 1
        self.grid_h = self.df['row_index'].max() + 1
        
        self.base_spread_prob = 0.01
        self.burn_out_prob = 0.001
        self.tree_weights = {1: 1.5, 3: 1.25, 4: 1.1, 2: 1.0, 0: 0.0}
        
        mission_space = MissionSpace(mission_func=lambda: "extinguish all fires")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w + 2,
            height=self.grid_h + 2,
            max_steps=2000,
            render_mode=render_mode,
            see_through_walls=True, 
            agent_view_size=101      # 홀수로 수정 (반드시 3, 5, 7... 중 하나)
        )
        
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _gen_grid(self, width, height):
        # 1. 전달받은 width, height를 사용하여 그리드 생성
        self.grid = Grid(width, height)
        
        # 2. 실제 데이터 범위에 맞춰 벽 생성 (0부터 width, height까지)
        self.grid.wall_rect(0, 0, width, height)

        # 3. 나무 배치 (기존 로직 유지)
        self.tree_cells = []
        for _, row in self.df.iterrows():
            tx, ty = int(row['col_index']) + 1, int(row['row_index']) + 1
            if row['is_tree'] == 1:
                self.grid.set(tx, ty, HealthyTree(tree_type=row['tree_type']))
                self.tree_cells.append((tx, ty))
        
        self.initial_tree_count = len(self.tree_cells)
        self.agent_pos = (1, 1)
        self.agent_dir = 0

    def get_realtime_prob(self, x, y):
        # CSV 데이터 추출 (좌표 보정)
        row = self.df[(self.df['col_index'] == x-1) & (self.df['row_index'] == y-1)].iloc[0]
        
        # 1. 수목 가중치
        w_tree = self.tree_weights.get(row['tree_type'], 1.0)
        # 2. 경사도 가중치
        slope = row['slope']
        w_slope = 1.0 if slope < 5 else (1.5 if slope < 15 else 2.0)
        # 3. 바람 및 사면 가중치
        aspect = row['aspect']
        diff = abs(aspect - self.target_aspect)
        if diff > 180: diff = 360 - diff
        w_aspect = 1.5 if diff < 45 else (1.2 if diff < 90 else 1.0)
        
        prob = self.base_spread_prob * w_tree * w_slope * w_aspect
        return min(prob, 0.05) # 최대 0.05 제한

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 바람 설정 (0:북, 1:동, 2:남, 3:서 - 바람이 불어가는 방향 기준)
        self.wind_idx = self.np_random.integers(0, 4)
        wind_cfg = {
            0: {"vec": [0, 1], "target": 180}, # 북풍 -> 남쪽(180) 사면 위험
            1: {"vec": [-1, 0], "target": 270},
            2: {"vec": [0, -1], "target": 0},
            3: {"vec": [1, 0], "target": 90}
        }
        self.wind_vec = wind_cfg[self.wind_idx]["vec"]
        self.target_aspect = wind_cfg[self.wind_idx]["target"]

        # 초기 화재 발생 (5곳)
        fire_nodes = self.np_random.choice(len(self.tree_cells), 5, replace=False)
        for idx in fire_nodes:
            fx, fy = self.tree_cells[idx]
            p = self.get_realtime_prob(fx, fy)
            self.grid.set(fx, fy, BurningTree(spread_prob=p))

        # 에이전트 랜덤 배치 및 상태 초기화
        self.agent_pos = self.place_agent()
        self.ammo = 3
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = -0.01 # Time Penalty
        
        # 1. 에이전트 이동 (0:우, 1:하, 2:좌, 3:상)
        moves = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
        dm = moves[action]
        next_p = (max(1, min(self.grid_w, self.agent_pos[0] + dm[0])),
                  max(1, min(self.grid_h, self.agent_pos[1] + dm[1])))
        self.agent_pos = next_p

        # 2. 화재 진압 판정
        curr_obj = self.grid.get(*self.agent_pos)
        if isinstance(curr_obj, BurningTree):
            reward += 2.0 + (20 * curr_obj.spread_prob)
            self.grid.set(*self.agent_pos, ExtinguishedTree())
            self.ammo -= 1
            
            # 소화탄 소진 시 에이전트 교체 (새로운 랜덤 위치)
            if self.ammo <= 0:
                self.agent_pos = self.place_agent()
                self.ammo = 3

        # 3. 화재 확산 로직
        fire_list = [(x, y) for x in range(1, self.grid_w+1) for y in range(1, self.grid_h+1) 
                     if isinstance(self.grid.get(x, y), BurningTree)]
        
        spread_count = 0
        for fx, fy in fire_list:
            # 4방향 확산
            for dx, dy in [[0,1],[0,-1],[1,0],[-1,0]]:
                nx, ny = fx+dx, fy+dy
                neighbor = self.grid.get(nx, ny)
                if isinstance(neighbor, HealthyTree):
                    p = self.get_realtime_prob(nx, ny)
                    if random.random() < p:
                        self.grid.set(nx, ny, BurningTree(spread_prob=p))
                        spread_count += 1
            # 전소 판정
            if random.random() < self.burn_out_prob:
                self.grid.set(fx, fy, BurntTree())

        reward -= (spread_count * 1.0)

        # 4. 종료 조건 및 추가 보상
        current_fires = [(x, y) for x in range(1, self.grid_w+1) for y in range(1, self.grid_h+1) 
                         if isinstance(self.grid.get(x, y), BurningTree)]
        burnt_count = sum(1 for x in range(1, self.grid_w+1) for y in range(1, self.grid_h+1) 
                          if isinstance(self.grid.get(x, y), (BurntTree, BurningTree)))
        
        terminated = False
        truncated = False
        
        if not current_fires: # 성공
            terminated = True
            alive_trees = self.initial_tree_count - burnt_count
            reward += alive_trees * 5.0
        elif burnt_count >= self.initial_tree_count * 0.5: # 실패
            terminated = True
            reward -= 100.0

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        ax, ay = self.agent_pos
        wx, wy = self.wind_vec
        
        fires = [(x, y, obj.spread_prob) for x in range(1, self.grid_w+1) for y in range(1, self.grid_h+1) 
                 if isinstance((obj := self.grid.get(x, y)), BurningTree)]
        
        if not fires:
            return np.zeros(9, dtype=np.float32)

        # 1. 가장 가까운 화재셀 거리
        dists = [((x-ax)**2 + (y-ay)**2, x, y) for x, y, p in fires]
        closest = min(dists)
        
        # 2. 확산 확률이 가장 높은(위험한) 화재셀 거리
        riskiest = max(fires, key=lambda x: x[2])
        
        obs = np.array([
            ax, ay,           # 에이전트 위치
            wx, wy,           # 바람 벡터
            self.ammo,        # 소화탄 잔량
            closest[1] - ax, closest[2] - ay, # 최단거리 화재 (상대좌표)
            riskiest[0] - ax, riskiest[1] - ay  # 최고위험 화재 (상대좌표)
        ], dtype=np.float32)
        
        return obs