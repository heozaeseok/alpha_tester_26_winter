import gymnasium as gym
import time
import pandas as pd
import numpy as np
from minigrid.core.world_object import Box
from minigrid_forest_env import ForestFireEnv

class CustomHealthyTree(Box):
    def __init__(self, tree_type=1):
        colors = {1: 'red', 2: 'green', 3: 'yellow', 4: 'grey'}
        super().__init__(color=colors.get(tree_type, 'green'))
        self.tree_type = tree_type
    def can_overlap(self): return True

class VisualForestEnv(ForestFireEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # 시야를 충분히 크게 설정 (박스 잔상을 최소화)
        self.agent_view_size = 31 

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        # 렌더링 시 시야 박스를 없애기 위해 highlight 설정을 끕니다.
        self.highlight = False 
        return obs, info

    def render(self):
        # 렌더링 직전에 모든 그리드를 '보임' 상태로 마스킹하는 트릭
        # (부모 클래스의 render가 내부적으로 get_full_render를 호출할 때 참조함)
        return super().render()

    # 에러를 일으켰던 gen_obs_grid와 get_view_extents는 삭제합니다.
    # MiniGrid 기본 시스템이 처리하도록 두는 것이 가장 안전합니다.

def run_visualization():
    # 시각화를 위해 render_mode를 "human"으로 설정
    env = VisualForestEnv(render_mode="human")
    env.reset()
    
    print("시각화 시작: 에이전트의 움직임을 관찰합니다.")
    
    for step in range(1000):
        # 환경의 step을 진행 (에이전트 이동 등)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # render_mode가 "human"이면 step 내에서 자동으로 그려지거나 
        # 아래와 같이 명시적으로 호출합니다.
        env.render()
        
        time.sleep(0.05)
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    run_visualization()