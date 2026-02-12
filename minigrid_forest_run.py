import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
import os
import torch
import matplotlib.pyplot as plt
import minigrid_forest_env

# 1. 환경 등록
register(
    id="ForestFireMLP-v22",
    entry_point="minigrid_forest_env:ForestFireEnv",
)

# 2. 경로 및 파라미터 설정
BASE_PATH = r"C:\Users\CIL\Desktop\minigird_forest"
MODEL_DIR = os.path.join(BASE_PATH, "learned_model")
GRAPH_DIR = os.path.join(BASE_PATH, "reward_graph")
LOG_DIR = os.path.join(BASE_PATH, "logs")

MODEL_NAME = "ppo_forest_fire_v22"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

TOTAL_TIMESTEPS = 30000000 

# GPU 사용 가능 시 cuda, 아니면 cpu 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 결과 시각화 함수 (예외 처리 제거)
def plot_results(log_folder, save_folder, title="Learning Curve"):
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.3, label='Raw Reward', color='blue')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_folder, f"{title}.png")
    plt.savefig(save_path)
    print(f"[Info] Reward graph saved at: {save_path}")
    plt.close()

# 4. 학습 실행
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = gym.make("ForestFireMLP-v22")
    env = Monitor(env, LOG_DIR) 

    print(f"Using Device: {DEVICE}")
    print(f"Training Start... (Steps: {TOTAL_TIMESTEPS})")
    
    model = PPO("MlpPolicy", env, verbose=1, device=DEVICE)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    print("Training Finished!")
    model.save(FULL_MODEL_PATH)
    print(f"[Info] Model saved at: {FULL_MODEL_PATH}.zip")
    
    plot_results(LOG_DIR, GRAPH_DIR, title="ForestFire_Training_Result")

    env.close()