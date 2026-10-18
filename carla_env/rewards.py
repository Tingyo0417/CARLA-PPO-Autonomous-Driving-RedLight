import numpy as np
from config import CONFIG

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            # Keep this per environment/episode. A module-global timer can leak
            # a previous episode's stopped state into the next route.
            low_speed_timer = getattr(env, 'low_speed_timer', 0.0)

            # 如果車輛正在停等紅燈或剛過線，不要累積低速計時器，避免因正確停等或微過線而被誤判為「異常怠速」終止
            red_grace = getattr(env, 'red_light_grace_timer', 0.0)
            is_near_intersection = getattr(env, 'distance_to_red_light', 999.0) < 25.0
            speed = env.vehicle.get_speed()
            
            # 【加強保護】檢查是否有活躍的紅燈 (包含 last_traffic_light 仍為紅燈的情況)
            has_active_red = (hasattr(env, 'is_at_red_light') and env.is_at_red_light)
            if not has_active_red and hasattr(env, 'last_traffic_light') and env.last_traffic_light is not None:
                try:
                    import carla
                    if env.last_traffic_light.get_state() == carla.TrafficLightState.Red:
                        if getattr(env, 'distance_to_red_light', 999.0) < 20.0:
                            has_active_red = True
                except:
                    pass
            
            if has_active_red or red_grace > 0:
                low_speed_timer = 0.0
            elif speed >= 1.0:
                # This must be a consecutive low-speed timer. The previous
                # logic accumulated elapsed episode time even while driving,
                # so any later brief slowdown triggered Vehicle stopped.
                low_speed_timer = 0.0
            elif is_near_intersection:
                # Give the vehicle extra time to settle near an intersection.
                low_speed_timer += (1.0 / env.fps) * 0.33
            else:
                low_speed_timer += 1.0 / env.fps
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            env.low_speed_timer = low_speed_timer

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

        if getattr(env, 'collision_detected', False):
            env.terminal_state = True
            terminal_reason = "Collision"

        # 新增紅燈違規判斷
        if hasattr(env, 'violated_red_light') and env.violated_red_light:
            env.terminal_state = True
            terminal_reason = "Red light violation"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            env.low_speed_timer = 0.0
            # 如果是因為闖紅燈終止，給予更重的負分 (例如 penalty_reward * 2)
            current_penalty = penalty_reward * 2 if "Red light" in terminal_reason else penalty_reward
            
            # 【重罰不當行為】：對於衝出車道和惡意掛機給予毀滅性打擊，逼迫它回歸正途
            if "Off-track" in terminal_reason:
                current_penalty = -100
            elif "Vehicle stopped" in terminal_reason:
                current_penalty = -100
                
            reward += current_penalty
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        # 供評估記錄使用，方便區分卡住、偏離車道與紅燈終止。
        env.terminal_reason = terminal_reason
        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# Reward_fn5
def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # 2. 修改：紅燈煞車獎勵邏輯 (讓 AI 知道停下來是好事)
    idling_penalty = 0.0
    if hasattr(env, 'is_at_red_light') and env.is_at_red_light:
        dist_to_light = getattr(env, 'distance_to_red_light', 999.0)
        # 依照距離決定「允許的最高速度」
        target_stop_speed = max(0.0, dist_to_light - 5.0) 
        
        if speed_kmh <= target_stop_speed + 2.0:
            # Waiting is correct but must not pay the same reward as route
            # progress on green; otherwise long red-light dwell dominates the
            # return and makes stopping everywhere an attractive local optimum.
            speed_reward = 0.2
        else:
            # 車速降不下來，強烈減少獎勵
            speed_reward = max(0.0, 1.0 - (speed_kmh - target_stop_speed) / 10.0)
            
    # 【新增】防掛機機制：如果綠燈/平時，但車子完全不走，把分數扣成負的！
    # 這能解決「綠燈了它卻不肯起步」的問題，因為停在原地會不斷大扣分。
    elif speed_kmh < 1.0:
        speed_reward = 0.0      # 先將基礎速度獎勵歸零，避免把後面的正向乘數倒扣
        idling_penalty = 2.0    # 獨立扣分，避免破壞乘法邏輯

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # 額外的絕對扣分：偏離車中心過多
    deviation_penalty = 0.0
    if env.distance_from_center > 1.5:
        deviation_penalty = (env.distance_from_center - 1.5) * 5.0

    # Final reward
    # Combine multiplicatively but with base values so deviation doesn't completely kill rewards
    base_reward = speed_reward * (0.5 + 0.5 * centering_factor) * (0.5 + 0.5 * angle_factor) * (0.5 + 0.5 * distance_std_factor)
    
    # Reward actual displacement on green. This gives the first few metres
    # after a traffic light an immediate learning signal instead of waiting
    # for the speed term to recover.
    is_red_light = bool(getattr(env, 'is_at_red_light', False))
    progress = max(0.0, min(1.0, getattr(env, 'progress_since_last_step', 0.0)))
    progress_reward = 0.0 if is_red_light else 0.5 * progress

    # Penalize rapid steering changes, especially a left/right sign reversal.
    # This acts on the applied (rate-limited) control and remains small during
    # a legitimate smooth turn.
    current_steer = float(env.vehicle.control.steer)
    previous_steer = float(getattr(env, '_reward_prev_steer', current_steer))
    steer_change = abs(current_steer - previous_steer)
    speed_weight = max(0.0, min(1.0, speed_kmh / 10.0))
    steering_penalty = 0.15 * steer_change * speed_weight
    if (current_steer * previous_steer < 0.0
            and abs(current_steer) > 0.05
            and abs(previous_steer) > 0.05):
        steering_penalty += 0.1 * (abs(current_steer) + abs(previous_steer)) * speed_weight
    env._reward_prev_steer = current_steer

    # 計算最終成績
    reward = (base_reward + progress_reward
              - idling_penalty - deviation_penalty - steering_penalty)

    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_waypoints(env):
    """
        reward
            - Each time the vehicle overpasses a waypoint, it will receive a reward of 1.0
            - When the vehicle does not pass a waypoint, it receives a reward of 0.0
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)
