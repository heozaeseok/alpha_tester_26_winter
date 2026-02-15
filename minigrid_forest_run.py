import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.envs.registration import register
import os
import torch
import matplotlib.pyplot as plt
import minigrid_forest_env

# 1. 환경 등록 (V22 유지)
try:
    register(
        id="ForestFireMLP-v22",
        entry_point="minigrid_forest_env:ForestFireEnv",
    )
except:
    pass # 이미 등록된 경우 무시

# 2. 경로 설정 (사용자 경로에 맞게 자동 조정)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "learned_model")
GRAPH_DIR = os.path.join(BASE_PATH, "reward_graph")
LOG_DIR = os.path.join(BASE_PATH, "logs")

MODEL_NAME = "ppo_forest_fire_v22"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 학습 스텝 설정 (환경이 복잡하므로 최소 1M~5M 스텝 권장)
TOTAL_TIMESTEPS = 5_000_000 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 결과 시각화 함수
def plot_results(log_folder, save_folder, title="Learning Curve"):
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    import numpy as np
    
    results = load_results(log_folder)
    x, y = ts2xy(results, 'timesteps')
    
    if len(x) == 0:
        print("[Error] No data to plot.")
        return

    # 이동 평균 계산 (그래프 가독성 향상)
    def moving_average(values, window):
        weights = np.ones(window) / window
        return np.convolve(values, weights, 'valid')

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.2, color='blue', label='Raw Reward')
    
    if len(y) > 50:
        y_av = moving_average(y, 50)
        plt.plot(x[len(x)-len(y_av):], y_av, color='red', label='Moving Average (50)')

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

    # 중간 저장 콜백 (학습이 길어질 경우 대비)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=MODEL_DIR,
        name_prefix="forest_fire_model_checkpoint"
    )

    print(f"Using Device: {DEVICE}")
    print(f"Training Start... (Total Steps: {TOTAL_TIMESTEPS})")
    
    # 전략적 행동 유도를 위한 최적화 모수 설정
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=DEVICE,
        n_steps=4096,           # 한 번 업데이트 시 수집할 데이터 양 증가
        batch_size=256,         # 학습 안정성 향상
        n_epochs=10,            # 동일 데이터 반복 학습 횟수
        learning_rate=3e-4,     # 표준 학습률
        gamma=0.99,             # 미래 보상 중시
        gae_lambda=0.95,        # 어드밴티지 추정 편향 조절
        ent_coef=0.01,          # 초기 탐색 유도 (불 끄러 다니는 법 배우기)
        clip_range=0.2          # 업데이트 안정화
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    
    print("Training Finished!")
    model.save(FULL_MODEL_PATH)
    print(f"[Info] Model saved at: {FULL_MODEL_PATH}.zip")
    
    plot_results(LOG_DIR, GRAPH_DIR, title="ForestFire_V22_Result")

    env.close()