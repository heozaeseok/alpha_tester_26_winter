import gymnasium as gym
import time
# 위에서 정의한 환경 클래스가 포함된 파일명이 minigrid_forest_env.py라고 가정합니다.
from minigrid_forest_env import ForestFireEnv 

def visualize_env():
    # 1. 환경 생성 (render_mode를 human으로 설정하여 창을 띄웁니다)
    env = ForestFireEnv(render_mode="human")
    
    # 2. 환경 초기화
    # reset 시 내부적으로 CSV를 읽고 tree_type에 맞는 HealthyTree를 배치합니다.
    obs, info = env.reset()
    
    print("환경 시각화를 시작합니다. (창을 닫으려면 터미널에서 Ctrl+C)")
    
    try:
        for step in range(1000):
            # 3. 랜덤 액션 샘플링 (0:우, 1:하, 2:좌, 3:상)
            action = env.action_space.sample()
            
            # 4. Step 진행
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 5. 렌더링 (human 모드이므로 자동으로 창이 업데이트됩니다)
            env.render()
            
            # 시각화를 관찰하기 위한 짧은 대기
            time.sleep(0.1)
            
            if terminated or truncated:
                print(f"에피소드 종료. 최종 보상: {reward}")
                break
                
    except KeyboardInterrupt:
        print("사용자에 의해 종료되었습니다.")
    finally:
        env.close()

if __name__ == "__main__":
    visualize_env()