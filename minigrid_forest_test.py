import gymnasium as gym
from stable_baselines3 import PPO
import time
import os
import minigrid_forest_env # 환경 등록 필수

# ==========================================
# [설정] 경로 및 파라미터
# ==========================================
BASE_PATH = r"C:\Users\CIL\Desktop\minigird_forest"
MODEL_PATH = os.path.join(BASE_PATH, "learned_model", "ppo_forest_fire_v22_ver4") # .zip 제외하고 경로 지정

# 학습 때와 동일한 환경 설정 권장

# ==========================================
# [메인] 테스트 실행
# ==========================================
if __name__ == "__main__":
    # 1. 모델 파일 존재 확인
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"[Error] Model file not found at: {MODEL_PATH}.zip")
        print("Please run the training script first.")
        exit()

    print(f"Loading Model from: {MODEL_PATH}")
    
    # 2. 모델 불러오기
    loaded_model = PPO.load(MODEL_PATH)
    
    # 3. 테스트 환경 생성 (Render Mode: Human)
    test_env = gym.make("ForestFireMLP-v22", render_mode="human")
    
    # 4. 테스트 루프 (5 에피소드)
    for episode in range(1, 6):
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        
        # 현재 불 개수 확인 (내부 변수 접근)
        fire_locs = [
            pos for pos in test_env.unwrapped.trees 
            if isinstance(test_env.unwrapped.grid.get(*pos), minigrid_forest_env.BurningTree)
        ]
        print(f"\n[Episode {episode}] Start! Initial Fires: {len(fire_locs)}")
        
        while not done:
            # 행동 예측 (Deterministic=True: 학습된 대로만 행동)
            action, _ = loaded_model.predict(obs, deterministic=True)
            
            # 환경 진행
            obs, reward, terminated, truncated, _ = test_env.step(action)
            
            # 렌더링 및 딜레이
            test_env.render()
            total_reward += reward
            time.sleep(0.05) # 속도 조절

            if terminated or truncated:
                done = True
                status = "Success (All Clear!)" if terminated else "Fail (Timeout)"
                print(f"Result: {status} | Total Reward: {total_reward:.2f}")

    test_env.close()
    print("\nTest Finished.")