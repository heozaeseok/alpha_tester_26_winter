import gymnasium as gym
from gymnasium.envs.registration import register
import minigrid_forest_env
import time
import os

# 1. 환경 등록
try:
    register(
        id="ForestFireMLP-v22",
        entry_point="minigrid_forest_env:ForestFireEnv",
    )
except:
    pass

def run_random_agent(num_episodes=3):
    # 2. 환경 생성 (렌더링 모드: human)
    env = gym.make("ForestFireMLP-v22", render_mode="human")
    
    # 시야 하이라이트 제거 (전체 맵 깨끗하게 보기)
    env.unwrapped.highlight = False 

    print("--- 랜덤 에이전트 테스트 시작 (학습 X) ---")
    print("나무 위치가 이동되었는지, 불이 나무끼리만 번지는지 확인하세요.")

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step = 0
        total_reward = 0

        print(f"\n--- Episode {episode + 1} Start ---")

        while not (terminated or truncated):
            # [핵심] 학습된 모델 없이 '무작위' 행동 선택
            action = env.action_space.sample()
            
            # 환경 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # 시각화 갱신 및 속도 조절
            # 불이 번지는 것을 천천히 보려면 시간을 늘리세요 (예: 0.1)
            time.sleep(0.005) 
            env.render()

        print(f"Episode Finished | Steps: {step} | Total Reward: {total_reward:.2f}")
        time.sleep(1)

    env.close()
    print("테스트 종료.")

if __name__ == "__main__":
    run_random_agent(num_episodes=3)