import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import minigrid_forest_env
import time
import os
import numpy as np

# 1. 환경 등록 (절대 생략하면 안 됩니다)
try:
    register(
        id="ForestFireMLP-v22",
        entry_point="minigrid_forest_env:ForestFireEnv",
    )
except:
    # 이미 등록된 경우 에러 방지
    pass

def test_forest_fire(model_path, num_episodes=5):
    # 2. 환경 생성 (agent_view_size를 맵 전체를 덮을 만큼 크게 설정)
    env = gym.make("ForestFireMLP-v22", render_mode="human")
    
    # [핵심] 하얀 박스(하이라이트) 기능을 끕니다.
    env.unwrapped.highlight = False 

    # 3. 모델 로드
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return
        
    model = PPO.load(model_path)
    print("모델 로드 완료.")

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step = 0

        print(f"--- 에피소드 {episode + 1} 시작 ---")

        while not (terminated or truncated):
            # 하얀 박스 제거 상태 유지
            env.unwrapped.highlight = False 
            
            # 행동 예측
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray): 
                action = action.item()
            
            # 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            
            step += 1
            # 렌더링 속도 (0.001초 대기)
            time.sleep(0.001) 
            env.render()

        print(f"에피소드 종료 | 총 스텝: {step}")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    # 경로를 다시 한번 확인해 주세요.
    FINAL_MODEL_PATH = r"C:\Users\USER\Desktop\forest_fire\ppo_forest_fire_parallel_final.zip"
    
    test_forest_fire(FINAL_MODEL_PATH, num_episodes=5)