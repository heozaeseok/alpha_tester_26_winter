import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor # 로그 기록용 래퍼
import os
import numpy as np
import matplotlib.pyplot as plt # 그래프 그리기용
import minigrid_forest_env  # 환경 등록

# ==========================================
# [설정] 경로 및 파라미터
# ==========================================
# 1. 경로 설정
BASE_PATH = r"C:\Users\CIL\Desktop\minigird_forest"
MODEL_DIR = os.path.join(BASE_PATH, "learned_model")
GRAPH_DIR = os.path.join(BASE_PATH, "reward_graph")
LOG_DIR = os.path.join(BASE_PATH, "logs") # 학습 로그(CSV) 임시 저장소

MODEL_NAME = "ppo_forest_fire_v22"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 2. 환경 및 학습 파라미터
TOTAL_TIMESTEPS = 30000000 
DEVICE = 'cpu'

# ==========================================
# [함수] 학습 결과 그래프 그리기
# ==========================================
def plot_results(log_folder, save_folder, title="Learning Curve"):
    """
    Monitor로 저장된 log.csv 파일을 읽어서 보상 그래프를 그립니다.
    """
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    # 로그 데이터 로드
    # load_results는 Monitor가 저장한 csv를 읽어옵니다.
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    
    if len(x) > 0:
        plt.figure(figsize=(10, 5))
        
        # 원본 데이터 (투명하게)
        plt.plot(x, y, alpha=0.3, label='Raw Reward', color='blue')
    
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        save_path = os.path.join(save_folder, f"{title}.png")
        plt.savefig(save_path)
        print(f"[Info] Reward graph saved at: {save_path}")
        plt.close()

# ==========================================
# [메인] 학습 실행
# ==========================================
if __name__ == "__main__":
    # 1. 폴더 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. 환경 생성 및 Monitor 래핑
    # Monitor는 학습 데이터를 csv로 기록해줍니다 (그래프용)
    env = gym.make("ForestFireMLP-v22")
    env = Monitor(env, LOG_DIR) 

    print(f"Training Start... (Steps: {TOTAL_TIMESTEPS})")
    
    # 3. 모델 정의 및 학습
    model = PPO("MlpPolicy", env, verbose=1, device=DEVICE)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("Training Finished!")
    
    # 4. 모델 저장
    model.save(FULL_MODEL_PATH)
    print(f"[Info] Model saved at: {FULL_MODEL_PATH}.zip")
    
    # 5. 그래프 그리기 및 저장
    try:
        plot_results(LOG_DIR, GRAPH_DIR, title="ForestFire_Training_Result")
    except Exception as e:
        print(f"[Warning] Could not plot graph: {e}")
        print("Make sure 'matplotlib' and 'pandas' are installed.")

    env.close()