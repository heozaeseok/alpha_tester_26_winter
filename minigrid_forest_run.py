import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid_forest_env import ForestFireEnv

# 1. 환경 생성 (렌더링 모드 제거하여 속도 향상)
def make_env():
    return ForestFireEnv(render_mode=None)

if __name__ == "__main__":
    # 병렬 환경 사용 (학습 속도 대폭 향상)
    env = make_vec_env(make_env, n_envs=4)

    # 2. 최적화된 학습 파라미터 설정
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,      # 표준적인 시작점
        n_steps=2048,            # 업데이트 주기
        batch_size=128,          # 배치 사이즈 (50x50 맵 고려)
        n_epochs=10,             # 반복 학습 횟수
        gamma=0.99,              # 미래 보상 할인율
        gae_lambda=0.95,         # 어드밴티지 추정 계수
        ent_coef=0.01,           # 탐험(Exploration) 강제 (초기 길 찾기 중요)
        device="auto"
    )

    # 3. 학습 시작 (렌더링 없이 순수 연산)
    print("학습을 시작합니다...")
    model.learn(total_timesteps=3000000) # 최소 50만 스텝 권장

    # 4. 모델 저장
    model.save("ppo_forest_fire_v1")
    print("모델 저장 완료.")