import time
import gymnasium as gym
from stable_baselines3 import PPO
from minigrid_forest_env import ForestFireEnv

# 테스트 환경 파라미터
NUM_EPISODES = 10     # 변경 가능한 에피소드 수
DELAY_SECONDS = 0.005   # 시각적 확인을 위한 지연 시간

def main():
    # 1. 테스트를 위해 렌더링 모드를 켜서 환경 생성
    env = ForestFireEnv(render_mode="human")
    env.unwrapped.highlight = False 
    # 2. 학습 완료된 모델 불러오기
    model = PPO.load(r"C:\Users\USER\Desktop\forest_fire\ppo_forest_fire_v4.zip")
    print(f"모델 로드 완료. 총 {NUM_EPISODES}번의 에피소드 테스트를 시작합니다.")

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0
        
        while not done:
            time.sleep(DELAY_SECONDS)
            
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # [추가] 명시적으로 렌더링을 호출하여 화면을 갱신합니다.
            env.render() 
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
        print(f"에피소드 {episode + 1} 종료 | 스텝 수: {step_count} | 총 보상: {total_reward:.2f}")

    env.close()
    print("모든 테스트가 완료되었습니다.")

if __name__ == "__main__":
    main()