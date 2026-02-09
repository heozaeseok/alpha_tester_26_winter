# minigrid_forest_RL_PPO

# 🚁 Forest Fire Fighting Drone with RL (PPO)

https://github.com/user-attachments/assets/68a0af9a-5314-49f7-bdb8-63f2e9a4c56e

> **Optimizing Firefighting Drone Operations using Reinforcement Learning**

이 프로젝트는 강화학습(PPO)을 활용하여 제한된 자원(배터리, 소화탄)을 가진 드론이 산불 확산을 막고 화재를 진압하는 최적의 경로를 학습하는 시뮬레이션입니다. `Gymnasium`과 `MiniGrid`를 기반으로 커스텀 환경을 구축하였으며, `Stable-Baselines3` 라이브러리를 사용했습니다.

## 📌 Project Overview

- **Goal:** 산불의 확산을 최소화하고 가능한 많은 수목을 보존하는 것.
- **Key Challenges:**
  - **Stochastic Environment:** 화재는 확률적으로 확산(Spread)되거나 자연 소화(Burn-out)됩니다.
  - **Resource Constraints:** 드론은 배터리(50 step)와 소화탄(2개)의 제약을 가집니다.
  - **Strategic Decision:** 단순히 가까운 불이 아닌, 확산 위험도가 높은 불을 우선순위로 판단해야 합니다.
- **Algorithm:** Proximal Policy Optimization (PPO) with MLP Policy.

---

## 🌍 Environment Design (MiniGrid)

**Custom Grid World (24x24)**
맵은 3개의 구역(**Zone A, B, C**)으로 구성된 유기적인 숲 형태를 가집니다.

### 🌲 Objects
| Object | Description |
| :--- | :--- |
| **HealthyTree** (Green) | 정상 나무 (보존 대상) |
| **BurningTree** (Red) | 화재 발생 나무 (진압 대상) |
| **BurntTree** (Grey) | 전소된 나무 (장애물, 복구 불가) |
| **WaterTank** (Blue Key) | (1,1) 위치. 소화탄 보충 및 배터리 충전소 |
| **Stone** (Purple) | 불이 붙지 않는 장애물 |

### ⚙️ Dynamics
1. **Fire Spread:** 매 스텝 1%의 확률로 인접한 건강한 나무로 불이 옮겨붙습니다.
2. **Burn Out:** 매 스텝 0.1%의 확률로 나무가 전소되어 사라집니다.
3. **Restock Rule:** 드론은 물탱크 방문 시 배터리와 소화탄을 100% 충전합니다.

---

## 🧠 MDP Formulation

### 1. State Space (Observation)
에이전트는 10차원의 연속 벡터(Continuous Vector)를 관측합니다.

- `[0-1]` **Agent Pos:** 정규화된 좌표 $(x, y)$
- `[2]` **Water Level:** 소화탄 잔량 비율 (Current / Max)
- `[3-4]` **Nearest Fire Vec:** 가장 가까운 화재까지의 상대 거리 $(\Delta x, \Delta y)$
- `[5-6]` **High-Risk Fire Vec:** **가장 위험한(밀집된) 화재**까지의 상대 거리 $(\Delta x, \Delta y)$
- `[7-9]` **Zone Health Ratio:** Zone A, B, C 각각의 건강한 나무 비율 (거시적 상황 판단용)

### 2. Action Space (Discrete 5)
- `0`: Stay, `1`: Up, `2`: Right, `3`: Down, `4`: Left
- **Contact Extinguishment:** 화재가 있는 셀로 이동하면 자동으로 소화탄 1개를 소모하여 진압합니다.

### 3. Reward Function (Updated)
에이전트가 "숲 전멸 방지"와 "고위험 화재 우선 진압"을 학습하도록 설계되었습니다.

| Event | Reward Value | Description |
| :--- | :--- | :--- |
| **Extinguish** | `+2.0 + (Risk * 3.0)` | 주변에 번질 나무가 많은(Risk Score가 높은) 불을 끄면 가중치 부여 |
| **Success** | `Alive Trees * +5.0` | **모든 불을 껐을 때, 살려낸 나무 수에 비례하여 보상** |
| **Failure** | **`-100.0`** | **건강한 나무가 0개가 되면(숲 전멸) 즉시 종료 및 대량 페널티** |
| **Spread** | `-1.0` | 불이 확산될 때마다 페널티 (방치 파밍 전략 방지) |
| **Time Penalty** | `-0.01 / step` | 최단 경로 유도 |

---

## 🚀 Installation & Usage

### 1. Dependencies
Python 3.8+ 환경에서 아래 패키지들을 설치해야 합니다.

```bash
conda create -n rl_env python=3.8
conda activate rl_env
pip install gymnasium minigrid stable-baselines3 shimmy matplotlib numpy
