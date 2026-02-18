import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball
from minigrid.minigrid_env import MiniGridEnv
import pandas as pd
import random

# --- 객체 정의 (기존 유지) ---
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
        # CSV 경로는 사용자 환경에 맞춰 유지
        self.csv_path = r"/home/cvrp/heojaeseok/GOPT_cvrp/ps/subongsan_integrated_final.csv"
        self.df = pd.read_csv(self.csv_path)
        
        self.grid_w = self.df['col_index'].max() + 1    
        self.grid_h = self.df['row_index'].max() + 1
        
        self.base_spread_prob = 0.01
        self.burn_out_prob = 0.001
        self.tree_weights = {1: 1.5, 3: 1.25, 4: 1.1, 2: 1.0, 0: 0.0}
        
        # 최적화를 위한 좌표 추적 집합
        self.fire_coords = set() 
        self.burnt_coords = set()

        mission_space = MissionSpace(mission_func=lambda: "extinguish all fires")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_w + 2,
            height=self.grid_h + 2,
            max_steps=2000,
            render_mode=render_mode,
            see_through_walls=True, 
            agent_view_size=101
        )
        
        # 관측치 차원 수정: 9 -> 11 (바람 정렬도, 현재지점 확산가중치 추가)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.tree_cells = []
            for _, row in self.df.iterrows():
                tx, ty = int(row['col_index']) + 1, int(row['row_index']) + 1
                if row['is_tree'] == 1:
                    self.grid.set(tx, ty, HealthyTree(tree_type=row['tree_type']))
                    self.tree_cells.append((tx, ty))
            
            self.initial_tree_count = len(self.tree_cells)

            # --- 추가된 부분: 에이전트 초기 위치 설정 ---
            # MiniGrid의 기본 검증을 통과하기 위해 초기 위치를 잡아줍니다.
            # 실제 랜덤 배치는 reset의 self.place_agent()에서 다시 수행됩니다.
            self.agent_pos = (1, 1)
            self.agent_dir = 0

    def get_realtime_prob(self, x, y):
        # 경계 밖 예외 처리 (벽 등)
        if not (1 <= x <= self.grid_w and 1 <= y <= self.grid_h): return 0.0
        
        row = self.df[(self.df['col_index'] == x-1) & (self.df['row_index'] == y-1)].iloc[0]
        w_tree = self.tree_weights.get(row['tree_type'], 1.0)
        slope = row['slope']
        w_slope = 1.0 if slope < 5 else (1.5 if slope < 15 else 2.0)
        aspect = row['aspect']
        diff = abs(aspect - self.target_aspect)
        if diff > 180: diff = 360 - diff
        w_aspect = 1.5 if diff < 45 else (1.2 if diff < 90 else 1.0)
        
        return min(self.base_spread_prob * w_tree * w_slope * w_aspect, 0.1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fire_coords.clear()
        self.burnt_coords.clear()
        
        self.wind_idx = self.np_random.integers(0, 4)
        wind_cfg = {
            0: {"vec": [0, 1], "target": 180},
            1: {"vec": [-1, 0], "target": 270},
            2: {"vec": [0, -1], "target": 0},
            3: {"vec": [1, 0], "target": 90}
        }
        self.wind_vec = wind_cfg[self.wind_idx]["vec"]
        self.target_aspect = wind_cfg[self.wind_idx]["target"]

        fire_nodes = self.np_random.choice(len(self.tree_cells), 4, replace=False)
        for idx in fire_nodes:
            fx, fy = self.tree_cells[idx]
            p = self.get_realtime_prob(fx, fy)
            self.grid.set(fx, fy, BurningTree(spread_prob=p))
            self.fire_coords.add((fx, fy))

        self.agent_pos = self.place_agent()
        self.ammo = 3  # 소화탄 5개로 수정
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = -0.01 
        
        # 1. 에이전트 이동
        moves = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
        dm = moves[action]
        self.agent_pos = (max(1, min(self.grid_w, self.agent_pos[0] + dm[0])),
                          max(1, min(self.grid_h, self.agent_pos[1] + dm[1])))

        # 2. 화재 진압 판정
        curr_obj = self.grid.get(*self.agent_pos)
        if isinstance(curr_obj, BurningTree):
            reward += 5.0 + (30 * curr_obj.spread_prob)
            self.grid.set(*self.agent_pos, ExtinguishedTree())
            self.fire_coords.discard(self.agent_pos)
            self.ammo -= 1
            if self.ammo <= 0:
                self.agent_pos = self.place_agent()
                self.ammo = 3 # 리스폰 시 다시 5개

        # 3. 화재 확산 로직
        spread_count = 0
        new_fires = []
        extinguished_fires = []

        for fx, fy in list(self.fire_coords):
            curr_fire = self.grid.get(fx, fy)
            if not isinstance(curr_fire, BurningTree): continue

            for dx, dy in [[0,1],[0,-1],[1,0],[-1,0]]:
                nx, ny = fx+dx, fy+dy
                if isinstance(self.grid.get(nx, ny), HealthyTree):
                    p = self.get_realtime_prob(nx, ny)
                    if random.random() < p:
                        self.grid.set(nx, ny, BurningTree(spread_prob=p))
                        new_fires.append((nx, ny))
                        spread_count += 1
            
            if random.random() < self.burn_out_prob:
                self.grid.set(fx, fy, BurntTree())
                extinguished_fires.append((fx, fy))
                self.burnt_coords.add((fx, fy))

        for nf in new_fires: self.fire_coords.add(nf)
        for ef in extinguished_fires: self.fire_coords.remove(ef)

        reward -= (spread_count * 0.5)

        # 4. 종료 조건
        terminated = False
        if not self.fire_coords:
            terminated = True
            reward += (self.initial_tree_count - len(self.burnt_coords)) * 5.0
        elif len(self.burnt_coords) + len(self.fire_coords) >= self.initial_tree_count * 0.75:
            terminated = True
            reward -= 100.0

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        ax, ay = self.agent_pos
        wx, wy = self.wind_vec
        
        if not self.fire_coords:
            return np.zeros(11, dtype=np.float32)

        # 화재 데이터 정리
        fire_data = []
        for (fx, fy) in self.fire_coords:
            obj = self.grid.get(fx, fy)
            if isinstance(obj, BurningTree):
                fire_data.append((fx, fy, obj.spread_prob))

        if not fire_data: return np.zeros(11, dtype=np.float32)

        # 기존 관측치 계산 (최단거리 및 최고위험)
        dists = [((x-ax)**2 + (y-ay)**2, x, y) for x, y, p in fire_data]
        closest = min(dists)
        riskiest = max(fire_data, key=lambda x: x[2])
        
        # --- 신규 전략 상태 추가 ---
        # 1. 위험 화재 방향과 풍향의 일치도 (Alignment)
        rx, ry = riskiest[0] - ax, riskiest[1] - ay
        dist_norm = np.sqrt(rx**2 + ry**2) + 1e-6
        alignment = (rx/dist_norm * wx) + (ry/dist_norm * wy) # 내적
        
        # 2. 현재 위치의 확산 가중치 (Local Risk)
        local_prob = self.get_realtime_prob(ax, ay)
        
        obs = np.array([
            ax, ay, wx, wy, self.ammo, 
            closest[1]-ax, closest[2]-ay, 
            riskiest[0]-ax, riskiest[1]-ay,
            alignment,    # [9] 바람-화재 정렬도
            local_prob    # [10] 현재 지형 위험도
        ], dtype=np.float32)
        
        return obs