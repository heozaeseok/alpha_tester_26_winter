import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.envs.registration import register
import os
import torch
import minigrid_forest_env

# 1. 환경 등록 (V22)
try:
    register(
        id="ForestFireMLP-v22",
        entry_point="minigrid_forest_env:ForestFireEnv",
    )
except:
    pass

# 2. 경로 및 병렬 설정
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "learned_model")
LOG_DIR = os.path.join(BASE_PATH, "logs")

# --- 병렬 프로세스 수 지정 ---
NUM_ENVS = 8  # 본인의 CPU 코어 수에 맞춰 조절하세요 (예: 4, 8, 16)
# --------------------------

TOTAL_TIMESTEPS = 10_000_000 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env(rank, seed=0):
    """
    멀티프로세싱을 위한 환경 생성 함수
    """
    def _init():
        env = gym.make("ForestFireMLP-v22")
        # 각 환경마다 다른 시드를 부여하여 다양한 상황을 학습하게 함
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 3. 병렬 환경 생성
    # SubprocVecEnv는 각 환경을 별도의 프로세스에서 실행합니다.
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    
    # 벡터화된 환경 전용 모니터링 (로그 기록)
    env = VecMonitor(env, LOG_DIR)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1), 
        save_path=MODEL_DIR,
        name_prefix="ppo_forest_parallel"
    )

    print(f"Device: {DEVICE} | Parallel Envs: {NUM_ENVS}")
    print(f"Training Start... (Total Steps: {TOTAL_TIMESTEPS})")
    
    # 4. PPO 모델 설정 (병렬화에 최적화된 모수)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=DEVICE,
        n_steps=2048,           # 각 환경당 수집 스텝 (총 2048 * NUM_ENVS 만큼 수집 후 업데이트)
        batch_size=512,         # 병렬 데이터가 많으므로 배치 크기를 키워 학습 안정성 확보
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,          # 탐색 유지
        clip_range=0.2,
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    
    model.save(os.path.join(MODEL_DIR, "ppo_forest_fire_parallel_final"))
    print("Training Finished!")

    env.close()