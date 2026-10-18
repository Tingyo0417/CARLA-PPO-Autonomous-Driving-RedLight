# Reinforcement Learning for Autonomous Driving: PPO Lane Keeping and Red-Light Stopping

**English** | [中文](#中文)

Undergraduate capstone project, Department of Artificial Intelligence, Tamkang University (academic year 2025–26, advisor: Prof. Yin-Tien Wang). This repository contains the project's **reinforcement learning decision module** (author: Ting-Yo Chen). A PPO agent from Stable-Baselines3 is trained in the CARLA simulator to keep its lane, control its speed, and stop at red lights.

Full project report (Chinese): [docs/專題報告_強化學習在自駕車系統之應用.pdf](docs/專題報告_強化學習在自駕車系統之應用.pdf). Chapter 3 (sections 3.1–3.6) covers the PPO work.

<p align="center">
  <img width="80%" src="docs/videos/ppo_v10_eval_preview.gif">
  <br>
  <em>v10 evaluation clip: the car waits at an intersection based on CARLA's traffic-light state. Full video: <a href="docs/videos/ppo_v10_eval_demo.mp4">ppo_v10_eval_demo.mp4</a></em>
</p>

## My contributions

Built on top of [Maté's CARLA-SB3 training environment](https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment):

- **Multiplicative composite reward**: speed × lane centering × heading angle × lateral-offset stability. The spatial factors have a floor of 0.5, so one bad factor can't zero out the whole reward and stall learning.
- **Steering smoothing**: exponential smoothing on the action output (`action_smoothing`) to reduce steering oscillation.
- **Brake action with throttle/brake exclusion**: the action space goes from `[steer, throttle]` to `[steer, throttle, brake]`. Only the larger of throttle and brake is applied.
- **Red-light braking**: the agent reads the traffic-light state from CARLA. During training, each light the car approaches is forced to red with 30% probability for 4–8 s. The project adds red-light-running detection and a red-light-specific speed reward.

## Architecture

<p align="center">
  <img width="80%" src="docs/images/adc_drl_flow.png">
  <br><em>ADC-DRL flow: VAE image features are combined with other state attributes into a 97-dim state, and PPO outputs control actions</em>
</p>

The state $s_t$ at each step has 97 dimensions:

| Source | Dim | Content |
| --- | --- | --- |
| VAE visual features | 64 | Latent vector of the front camera image from the VAE encoder |
| Vehicle state | 3 | steer, throttle, speed |
| Navigation | 30 | Upcoming waypoints |

The actor and critic share a feature extractor. Each then has its own two fully connected layers (500 → 300, ReLU). The actor outputs the mean $\mu$ of a Gaussian, with a trainable standard deviation $\sigma$, and samples continuous control actions from it. The critic outputs the state value $V(s_t)$.

<p align="center">
  <img width="70%" src="docs/images/actor_critic.png">
  <br><em>Actor-critic networks</em>
</p>

### PPO update loop

<p align="center">
  <img width="85%" src="docs/images/ppo_update.png">
</p>

1. Interact with CARLA and store $(s_t, a_t, r_{t+1}, s_{t+1})$ in the rollout buffer. Continue until a collision, a violation, or the step limit.
2. Compute advantages with GAE: $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$, $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$.
3. Minimize the total loss $L = -L^{CLIP} + c_1 L^{VF} - c_2 H$ with clip $\epsilon = 0.2$ and $c_1 = 0.5$.
4. Backpropagate with Adam and gradient clipping. PPO is on-policy, so the buffer is cleared after each update.

## Reward design

The base reward is the product of four factors, in $[0, 1]$:

$$\text{Reward} = \text{Speed} \times \text{Centering} \times \text{Angle} \times \text{DistanceStd}$$

| Factor | Definition |
| --- | --- |
| Speed | $V \le V_{min}$: $V / V_{min}$; $V_{min} < V \le V_{target}$: 1.0; $V > V_{target}$: decreases linearly toward $V_{max}$ |
| Centering | $\max(1 - D / D_{max}, 0)$, $D_{max} = 3.0$ m |
| Angle | $\max(1 - \lvert\theta / \theta_{max}\rvert, 0)$, $\theta_{max} = 90^\circ$ |
| DistanceStd | Std. dev. $\sigma$ of lane offset over the last 30 steps: $\max(1 - \sigma / 0.4, 0)$ |

**Red light**: the target speed depends on the distance $D_t$ to the light: $V_{stop} = \max(0, D_t - 5)$. If $V \le V_{stop} + 2$, the speed reward is 1.0. Otherwise it is $\max(0, 1 - (V - V_{stop}) / 10)$. The agent also gets an idling penalty for standing still too long when the light isn't red.

**Episode termination**: collision, lane deviation beyond $D_{max}$, speeding, or running a red light (crossing the stop line above 2 km/h while the light is red).

Relevant code:

- Reward functions: [`carla_env/rewards.py`](carla_env/rewards.py)
- Red-light detection, forced red lights, throttle/brake exclusion: [`carla_env/envs/carla_route_env.py`](carla_env/envs/carla_route_env.py)
- State encoding (including `red_light`): [`carla_env/state_commons.py`](carla_env/state_commons.py)
- Experiment settings: `_CONFIG_16` in [`config.py`](config.py)

## Training settings (config 16)

| Parameter | Value |
| --- | --- |
| learning rate | 3e-5 → 1e-6 schedule |
| gamma / gae_lambda | 0.98 / 0.95 |
| clip_range / ent_coef | 0.2 / 0.005 |
| n_steps / batch_size / n_epochs | 1024 / 128 / 5 |
| net_arch | pi: [500, 300], vf: [500, 300] |
| VAE | `vae_64` |
| action_smoothing | 0.75 |
| Timesteps | 5,000,000 |

## Results

<p align="center">
  <img width="70%" src="docs/images/training_reward.png">
  <br><em>Total reward per episode (moving average), red-light training run</em>
</p>

<p align="center">
  <img width="80%" src="docs/images/red_light_stop.jpg">
  <br><em>Stopping at a red light: speed 0 km/h, throttle and steering at zero, brake applied</em>
</p>

The repo includes two 5M-step runs. Each one has `config.json`, `progress.csv`, TensorBoard events, the model, and evaluation CSVs and plots:

| Folder | Description | Evaluation |
| --- | --- | --- |
| [`tensorboard/PPO_1776260506_id1`](tensorboard/PPO_1776260506_id1) | Baseline PPO (lane keeping + speed control) | 4/4 episodes succeeded, mean speed 18.39 km/h, mean lane offset 0.033 m |
| [`tensorboard/PPO_1785232694_id16`](tensorboard/PPO_1785232694_id16) | PPO + brake + red-light training | With the v10 control pipeline: 8/8 succeeded in both `final_v10` and `controller_v10_acceptance_8x4500` |

Notes:

- The figures in section 3.5 of the report (mean speed 22.85 km/h, 681.53 m driven, mean step reward 0.8959) come from a different test run. They don't match the CSVs in this repo one-to-one.
- The 8/8 success for id16 includes the effect of the external control pipeline, so it can't be attributed to PPO alone. The same model evaluated without the v10 pipeline (`model_5000000_steps_eval*`) failed all 20 episodes. That run is kept for comparison.
- Red-light stopping uses the traffic-light state provided directly by CARLA. Camera-based light recognition isn't implemented yet. Performance at other intersections and under other traffic conditions hasn't been tested.

## Installation and usage

The report's environment: CARLA 0.9.16, Stable-Baselines3 2.9.0, CUDA 13.0. Hardware: Ryzen 9 7900X3D, 2 × RTX 4090, 128 GB RAM. `requirements.txt` keeps the original framework's pinned versions. Adjust the `carla` package to match your CARLA version.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export CARLA_ROOT=<path to CARLA>
```

Train, inspect, and evaluate:

```bash
python train.py --config 16 --total_timesteps 5000000
tensorboard --logdir tensorboard
python eval.py --config 16 --model tensorboard/PPO_1785232694_id16/model_5000000_steps.zip
```

Evaluation routes are set in `eval_routes` in `carla_env/envs/carla_route_env.py`. For VAE training and data collection, see `vae/train_vae.py` and `carla_env/envs/collect_data_*.py`.

## Team

| Member | Module |
| --- | --- |
| Ting-Yo Chen | Reinforcement learning decision module (PPO, reward design, braking and red-light training) ← this repo |
| Cheng-Lun Chuang | VAE visual compression, LaneNet lane segmentation |
| Tzu-Yang Hsu | Vision-based autonomous emergency braking (AEB), system integration |

## Acknowledgements

The training environment is built on Alberto Maté's [CARLA-SB3-RL-Training-Environment](https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment), and the architecture follows Razak's ADC-DRL. If you use the original framework, please cite:

```
@software{Mate_CARLA-SB3-RL-Training-Environment_out-of-the-box_training_2023,
  author = {Mate, Alberto},
  month = jun,
  title = {{CARLA-SB3-RL-Training-Environment: out-of-the-box training and evaluation environment for DRL in CARLA simulator}},
  url = {https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment},
  version = {1.0.0},
  year = {2023}
}
```

---

<a id="中文"></a>

# 強化學習在自駕車系統之應用：PPO 車道保持與紅燈煞停

[English](#reinforcement-learning-for-autonomous-driving-ppo-lane-keeping-and-red-light-stopping) | **中文**

淡江大學人工智慧學系 114 學年度專題（指導老師：王銀添 教授）。本 repo 是專題中的 **強化學習決策模組**（負責人：陳廷祐），以 CARLA 模擬器搭配 Stable-Baselines3 的 PPO，訓練能車道置中、控制車速並在紅燈前停車的自駕車代理人。

完整專題報告：[docs/專題報告_強化學習在自駕車系統之應用.pdf](docs/專題報告_強化學習在自駕車系統之應用.pdf)（PPO 相關內容在第參章 3.1–3.6 節）。

<p align="center">
  <img width="80%" src="docs/videos/ppo_v10_eval_preview.gif">
  <br>
  <em>v10 評估片段：依 CARLA 號誌狀態於路口停等。完整影片：<a href="docs/videos/ppo_v10_eval_demo.mp4">ppo_v10_eval_demo.mp4</a></em>
</p>

## 我負責的部分

在 [Maté 的 CARLA-SB3 訓練環境](https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment) 上做了以下修改：

- **複合型乘法獎勵**：速度 × 車道置中 × 車身角度 × 歷史偏移穩定度，空間因子下限縮放至 0.5，避免單一因素讓總分歸零、模型停滯不前。
- **方向盤平滑**：對動作輸出做指數平滑（`action_smoothing`），減少轉向震盪。
- **煞車動作與油煞互斥**：動作空間由 `[steer, throttle]` 擴充為 `[steer, throttle, brake]`，並且油門與煞車只保留較大者。
- **紅燈煞車**：讀取 CARLA 號誌狀態，訓練時以 30% 機率把遇到的號誌強制設為紅燈（持續 4–8 秒），並加入闖紅燈判定與紅燈專用的速度獎勵。

## 系統架構

<p align="center">
  <img width="80%" src="docs/images/adc_drl_flow.png">
  <br><em>ADC-DRL 資訊流程：VAE 影像特徵與其他狀態合併成 97 維狀態，交給 PPO 輸出控制動作</em>
</p>

每個時間步的狀態 $s_t$ 由三部分組成，共 97 維：

| 來源 | 維度 | 內容 |
| --- | --- | --- |
| VAE 視覺特徵 | 64 | 前視相機影像經 VAE Encoder 壓縮的潛在向量 |
| 車輛狀態 | 3 | steer、throttle、speed |
| 導航路徑 | 30 | 前方 waypoints |

Actor 與 Critic 共用特徵抽取器，各自接兩層全連接（500 → 300，ReLU）。Actor 輸出高斯分佈的均值 $\mu$ 與可訓練的標準差 $\sigma$，抽樣出連續控制動作；Critic 輸出狀態價值 $V(s_t)$。

<p align="center">
  <img width="70%" src="docs/images/actor_critic.png">
  <br><em>Actor-Critic 網路</em>
</p>

### PPO 更新流程

<p align="center">
  <img width="85%" src="docs/images/ppo_update.png">
</p>

1. 與 CARLA 互動，把 $(s_t, a_t, r_{t+1}, s_{t+1})$ 存進 Rollout Buffer，直到碰撞、違規或達到最大步數。
2. 以 GAE 計算優勢值：$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$，$\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$。
3. 最小化總損失 $L = -L^{CLIP} + c_1 L^{VF} - c_2 H$，其中 clip $\epsilon = 0.2$、$c_1 = 0.5$。
4. Adam 反向傳播並做梯度裁剪；PPO 為 on-policy，更新後清空 buffer 再進入下一輪。

## 獎勵設計

基礎獎勵為四個因子相乘，值域 $[0, 1]$：

$$\text{Reward} = \text{Speed} \times \text{Centering} \times \text{Angle} \times \text{DistanceStd}$$

| 因子 | 計算方式 |
| --- | --- |
| Speed | $V \le V_{min}$：$V / V_{min}$；$V_{min} < V \le V_{target}$：1.0；$V > V_{target}$：線性遞減至 $V_{max}$ |
| Centering | $\max(1 - D / D_{max}, 0)$，$D_{max} = 3.0$ m |
| Angle | $\max(1 - \lvert\theta / \theta_{max}\rvert, 0)$，$\theta_{max} = 90^\circ$ |
| DistanceStd | 過去 30 步車道偏移標準差 $\sigma$：$\max(1 - \sigma / 0.4, 0)$ |

**紅燈情境**：依車輛與號誌距離 $D_t$ 設定目標速度 $V_{stop} = \max(0, D_t - 5)$。若 $V \le V_{stop} + 2$ 則速度獎勵為 1.0，否則為 $\max(0, 1 - (V - V_{stop}) / 10)$。另外非紅燈長時間停滯會給怠速懲罰。

**終止條件**：碰撞、偏離車道超過 $D_{max}$、超速，或紅燈時以高於 2 km/h 的速度通過停止線（闖紅燈）。

相關程式：

- 獎勵函數：[`carla_env/rewards.py`](carla_env/rewards.py)
- 紅燈偵測、強制紅燈、油煞互斥：[`carla_env/envs/carla_route_env.py`](carla_env/envs/carla_route_env.py)
- 狀態編碼（含 `red_light`）：[`carla_env/state_commons.py`](carla_env/state_commons.py)
- 實驗設定：[`config.py`](config.py) 的 `_CONFIG_16`

## 訓練設定（config 16）

| 參數 | 值 |
| --- | --- |
| learning rate | 3e-5 → 1e-6 排程 |
| gamma / gae_lambda | 0.98 / 0.95 |
| clip_range / ent_coef | 0.2 / 0.005 |
| n_steps / batch_size / n_epochs | 1024 / 128 / 5 |
| net_arch | pi: [500, 300]、vf: [500, 300] |
| VAE | `vae_64` |
| action_smoothing | 0.75 |
| 訓練步數 | 5,000,000 |

## 實驗結果

<p align="center">
  <img width="70%" src="docs/images/training_reward.png">
  <br><em>紅燈訓練版本的每回合總獎勵（移動平均）</em>
</p>

<p align="center">
  <img width="80%" src="docs/images/red_light_stop.jpg">
  <br><em>紅燈前煞停：速度 0 km/h，油門、方向盤歸零，煞車作動</em>
</p>

repo 內保留兩組 500 萬步的實驗，每組都有 `config.json`、`progress.csv`、TensorBoard events、模型與評估 CSV／圖：

| 資料夾 | 說明 | 評估結果 |
| --- | --- | --- |
| [`tensorboard/PPO_1776260506_id1`](tensorboard/PPO_1776260506_id1) | 一般 PPO（車道保持＋速度控制） | 4/4 回合成功，平均車速 18.39 km/h，平均車道偏移 0.033 m |
| [`tensorboard/PPO_1785232694_id16`](tensorboard/PPO_1785232694_id16) | PPO＋煞車＋紅燈訓練 | 搭配 v10 控制流程：`final_v10`、`controller_v10_acceptance_8x4500` 各 8/8 成功 |

說明：

- 報告第 3.5 節的數據（平均車速 22.85 km/h、行駛 681.53 m、平均每步獎勵 0.8959）來自另一次測試紀錄，無法與這裡的 CSV 一一對應。
- id16 的 8/8 成功包含外部控制流程的作用，不能全部歸因於 PPO；同一模型不加 v10 控制時的原始評估（`model_5000000_steps_eval*`）20 回合皆未成功，一併保留作比較。
- 紅燈停車使用的是 CARLA 直接提供的號誌狀態，尚未接攝影機號誌辨識；換路口、換交通條件後的表現仍待測試。

## 安裝與使用

報告中使用的環境：CARLA 0.9.16、Stable-Baselines3 2.9.0、CUDA 13.0；硬體為 Ryzen 9 7900X3D、RTX 4090 × 2、128 GB RAM。`requirements.txt` 沿用原框架的版本清單，請依自己的 CARLA 版本調整 `carla` 套件。

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export CARLA_ROOT=<CARLA 安裝路徑>
```

訓練、檢視與評估：

```bash
python train.py --config 16 --total_timesteps 5000000
tensorboard --logdir tensorboard
python eval.py --config 16 --model tensorboard/PPO_1785232694_id16/model_5000000_steps.zip
```

評估路線可在 `carla_env/envs/carla_route_env.py` 的 `eval_routes` 修改。VAE 訓練與資料收集請見 `vae/train_vae.py`、`carla_env/envs/collect_data_*.py`。

## 專題分工

| 成員 | 模組 |
| --- | --- |
| 陳廷祐 | 強化學習決策模組（PPO、獎勵設計、煞車與紅燈訓練）← 本 repo |
| 莊正綸 | VAE 視覺資訊壓縮、LaneNet 車道分割 |
| 徐子洋 | 純視覺 AEB 自動緊急煞車系統、雙系統整合 |

## 致謝

本專案的訓練環境建立在 Alberto Maté 的 [CARLA-SB3-RL-Training-Environment](https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment) 之上，架構參考 Razak 的 ADC-DRL。引用格式請見英文版的 [Acknowledgements](#acknowledgements)。
