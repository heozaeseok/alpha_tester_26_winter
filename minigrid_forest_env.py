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
        colors = {1: 'red', 2: 'green', 3: 'yellow', 4: 'grey'}
        super().__init__(color=colors.get(tree_type, 'green'))
        self.tree_type = tree_type
    def can_overlap(self): return True

class BurningTree(Ball):
    def __init__(self, prob=0.01): 
        super().__init__(color='red')
        self.spread_prob = prob
    def can_overlap(self): return True

class ForestFireEnv(MiniGridEnv):
    def __init__(self, render_mode="human"):
        # --- [환경 설정 모수 (Hyperparameters)] ---
        self.grid_w, self.grid_h = 50, 50
        self.shift_y = 4            # 나무 위치를 아래로 내리는 오프셋
        self.base_prob = 0.01       # 기본 확산 확률 (모든 확률 계산의 기초)
        self.ammo_limit = 3         # 에이전트가 가진 소화탄 개수
        self.max_steps = 1000       # 에피소드 최대 스텝
        
        # 수목 종류별 확산 가중치 (나무 타입에 따른 연소 속도 차이)
        # 1:침엽수(빠름), 2:활엽수(느림), 3:혼효림(중간), 4:기타
        self.tree_weights = {0: 0.0, 1: 1.5, 2: 0.8, 3: 1.2, 4: 1.0}
        
        # 지형 가중치 계수
        self.slope_norm = 45.0      # 경사도 영향력 조절 (기울기/45도 기준으로 가중치 부여)
        self.wind_impact = 3.0      # 바람 방향과 사면향이 일치할 때의 최대 가중치
        # ------------------------------------------

        self.df = pd.read_csv("subongsan_integrated_final.csv")
        self.terrain_data = {}
        self.initial_tree_count = 0
        
        for _, row in self.df.iterrows():
            tx = int(row['col_index']) + 1
            ty = int(row['row_index']) + 1 + self.shift_y # [수정] 4칸 아래로 이동
            
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
        self.fire_coords = set()
        self.ammo = self.ammo_limit
        
        tree_positions = []
        for (tx, ty), data in self.terrain_data.items():
            if data['is_tree'] == 1:
                self.grid.set(tx, ty, HealthyTree(tree_type=data['tree_type']))
                tree_positions.append((tx, ty))

        self.agent_pos = self._get_random_safe_pos()
        self.agent_dir = 0
        
        # 바람 설정 및 이진 관측값 매핑
        self.wind_mode = random.choice(['N', 'S', 'E', 'W'])
        wind_obs_map = {'N': (0,0), 'S': (1,0), 'E': (0,1), 'W': (1,1)}
        self.wind_obs = wind_obs_map[self.wind_mode]
        
        if len(tree_positions) >= 2:
            for fx, fy in random.sample(tree_positions, 2):
                p = self.calculate_spread_prob(fx, fy)
                self.grid.set(fx, fy, BurningTree(prob=p))
                self.fire_coords.add((fx, fy))

    def calculate_spread_prob(self, x, y):
        """
        각 셀의 확산 확률을 계산합니다.
        영향 요소: 수목 종류(tree_weights), 경사도(slope), 바람-사면향(aspect)
        """
        data = self.terrain_data.get((x, y), None)
        if not data or data['is_tree'] == 0: return 0.0
        
        # 1. 수목 가중치: 불에 잘 타는 수종일수록 확률 증가
        tw = self.tree_weights.get(data['tree_type'], 1.0)
        
        # 2. 경사도 가중치: 가파른 경사일수록 불길이 위로 빠르게 치솟음
        sw = 1.0 + (data['slope'] / self.slope_norm)
        
        # 3. 바람-사면향 가중치: 바람이 불어오는 방향과 경사면의 방향이 일치하면 산소 공급과 열 전달이 극대화됨
        wind_deg = {'N': 0, 'S': 180, 'E': 90, 'W': 270}[self.wind_mode]
        diff = abs(data['aspect'] - wind_deg)
        angle_diff = min(diff, 360-diff)
        # 차이가 0도에 가까울수록 self.wind_impact(2.0)에 가까운 값 생성
        aw = self.wind_impact - (angle_diff / 180.0) 
        
        return self.base_prob * tw * sw * aw

    def _get_random_safe_pos(self):
        while True:
            rx, ry = random.randint(1, self.grid_w), random.randint(1, self.grid_h)
            if not isinstance(self.grid.get(rx, ry), (HealthyTree, BurningTree)):
                return (rx, ry)
            
    def reset(self, seed=None, options=None):
        # 1. 부모 클래스(MiniGridEnv)의 reset을 호출하여 그리드 생성
        super().reset(seed=seed, options=options)
        
        # 2. MiniGrid 기본 관측값(Dict) 대신, 우리가 정의한 9개짜리 배열 반환
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        reward = -0.1 
        
        d_close_pre, d_danger_pre = self._get_fire_distances()

        moves = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}
        dx, dy = moves[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        if 0 < nx <= self.grid_w and 0 < ny <= self.grid_h:
            self.agent_pos = (nx, ny)

        d_close_post, d_danger_post = self._get_fire_distances()
        if d_close_post < d_close_pre: reward += 0.01
        if d_danger_post < d_danger_pre: reward += 0.01

        curr_obj = self.grid.get(*self.agent_pos)
        if isinstance(curr_obj, BurningTree):
            # 진압 보상: 확산 확률이 높은 셀일수록 더 큰 점수
            reward += 5.0 + (curr_obj.spread_prob * 10.0)
            self.grid.set(*self.agent_pos, Floor())
            self.fire_coords.discard(self.agent_pos)
            self.ammo -= 1
            if self.ammo <= 0: # 탄약 소진 시 랜덤 리스폰
                self.agent_pos = self._get_random_safe_pos()
                self.ammo = self.ammo_limit

        spread_count = self._spread_fire()
        reward -= (spread_count * 1.0) # 불이 번질 때마다 페널티

        terminated = len(self.fire_coords) == 0
        truncated = self.step_count >= self.max_steps
        
        # 현재 살아있는 나무 수 확인
        current_trees = sum(1 for (tx, ty) in self.terrain_data if isinstance(self.grid.get(tx, ty), HealthyTree))
        
        # 종료 조건: 나무의 절반 이상 소실
        if current_trees < (self.initial_tree_count * 0.5):
            terminated = True
            reward -= 100.0
        
        # 성공 종료: 남은 나무 수만큼 점수 가산
        if terminated and len(self.fire_coords) == 0:
            reward += current_trees * 5.0

        return self._get_obs(), reward, terminated, truncated, {}

    def _spread_fire(self):
        new_fires = set()
        for fx, fy in list(self.fire_coords):
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = fx + dx, fy + dy
                if (nx, ny) in self.terrain_data and isinstance(self.grid.get(nx, ny), HealthyTree):
                    prob = self.calculate_spread_prob(nx, ny)
                    if random.random() < prob:
                        new_fires.add((nx, ny))
        
        for nx, ny in new_fires:
            p = self.calculate_spread_prob(nx, ny)
            self.grid.set(nx, ny, BurningTree(prob=p))
            self.fire_coords.add((nx, ny))
        return len(new_fires)

    def _get_fire_distances(self):
        if not self.fire_coords: return 0, 0
        ax, ay = self.agent_pos
        dists = [np.sqrt((fx-ax)**2 + (fy-ay)**2) for fx, fy in self.fire_coords]
        
        danger_list = [(self.grid.get(fx, fy).spread_prob, fx, fy) for fx, fy in self.fire_coords]
        _, dfx, dfy = max(danger_list)
        dist_danger = np.sqrt((dfx-ax)**2 + (dfy-ay)**2)
        
        return min(dists), dist_danger

    def _get_obs(self):
        ax, ay = self.agent_pos
        if not self.fire_coords: return np.zeros(9, dtype=np.float32)

        dists = [((fx-ax)**2 + (fy-ay)**2, fx, fy) for fx, fy in self.fire_coords]
        _, cfx, cfy = min(dists)
        
        danger_list = [(self.grid.get(fx, fy).spread_prob, fx, fy) for fx, fy in self.fire_coords]
        _, dfx, dfy = max(danger_list)

        return np.array([
            ax, ay, 
            self.wind_obs[0], self.wind_obs[1],
            self.ammo,
            cfx - ax, cfy - ay,
            dfx - ax, dfy - ay
        ], dtype=np.float32)