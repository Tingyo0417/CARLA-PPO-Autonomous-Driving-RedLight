import torch
from torchvision import transforms

import os
from vae.models import VAE
import numpy as np
import gym
from config import CONFIG

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector

torch.cuda.empty_cache()


def load_vae(vae_dir, latent_size):
    model_dir = os.path.join(vae_dir, 'best.tar')
    model = VAE(latent_size)
    if os.path.exists(model_dir):
        state = torch.load(model_dir)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        return model
    raise Exception("Error - VAE model does not exist")


def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess(frame).unsqueeze(0)
    return frame


def create_encode_state_fn(vae, measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "angle_next_waypoint" in measurements_to_include,
                     "maneuver" in measurements_to_include,
                     "waypoints" in measurements_to_include,
                     "rgb_camera" in measurements_to_include,
                     "seg_camera" in measurements_to_include,
                     "end_wp_vector" in measurements_to_include,
                     "end_wp_fixed" in measurements_to_include,
                     "distance_goal" in measurements_to_include,
                     "red_light" in measurements_to_include]

    def create_observation_space():
        observation_space = {}
        if vae: observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(LSIZE, ), dtype=np.float32)
        low, high = [], []
        if measure_flags[0]: low.append(-1), high.append(1)
        if measure_flags[1]: low.append(0), high.append(1)
        if measure_flags[2]: low.append(0), high.append(2.0) # speed 縮放 (/60)
        if measure_flags[3]: low.append(-1.0), high.append(1.0) # angle 縮放 (/3.14)
        if measure_flags[11]: low.append(0), high.append(1)
        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        if measure_flags[4]: observation_space['maneuver'] = gym.spaces.Discrete(4)

        if measure_flags[5]: observation_space['waypoints'] = gym.spaces.Box(low=-5.0, high=5.0, shape=(15, 2),
                                                                             dtype=np.float32)

        if measure_flags[6]: observation_space['rgb_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        if measure_flags[7]: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        if measure_flags[8]: observation_space['end_wp_vector'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2), dtype=np.float32)
        if measure_flags[9]: observation_space['end_wp_fixed'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2), dtype=np.float32)
        if measure_flags[10]: observation_space['distance_goal'] = gym.spaces.Box(low=0, high=1500, shape=(1, 1), dtype=np.float32)

        return gym.spaces.Dict(observation_space)

    def encode_state(env):
        encoded_state = {}
        if vae:
            vae.eval()  # Ensure VAE is in eval mode before encoding
            with torch.no_grad():
                frame = preprocess_frame(env.observation)
                mu, logvar = vae.encode(frame)
                # Evaluation must be deterministic. Sampling with
                # reparameterize() injects fresh Gaussian noise into every
                # observation, so model.predict(..., deterministic=True) is
                # still stochastic and can make the vehicle drift off track.
                # Keep stochastic sampling during training, but use the
                # posterior mean for evaluation.
                if getattr(env, 'eval', False):
                    vae_latent = mu[0].cpu().detach().numpy().squeeze()
                else:
                    vae_latent = vae.reparameterize(mu, logvar)[0].cpu().detach().numpy().squeeze()
            encoded_state['vae_latent'] = vae_latent
        vehicle_measures = []
        if measure_flags[0]: vehicle_measures.append(env.vehicle.control.steer)
        if measure_flags[1]: vehicle_measures.append(env.vehicle.control.throttle)
        if measure_flags[2]: vehicle_measures.append(env.vehicle.get_speed() / 60.0) # 標準化
        if measure_flags[3]: vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint) / 3.14) # 標準化
        if measure_flags[11]: 
            is_red = hasattr(env, 'is_at_red_light') and env.is_at_red_light
            if is_red:
                dist = getattr(env, 'distance_to_red_light', 30.0)
                capped_dist = max(5.0, min(30.0, dist))
                val = 1.0 - ((capped_dist - 5.0) / 25.0)
                # 【修復】: 移除 max(0.1, val)，因為如果底層鎖死 0.1，AI 永遠無法分辨「很遠的紅燈」和「這不是紅燈 (0.0)」的界線
                vehicle_measures.append(max(0.0, val))
            else:
                vehicle_measures.append(0.0)
        encoded_state['vehicle_measures'] = vehicle_measures
        if measure_flags[4]: encoded_state['maneuver'] = env.current_road_maneuver.value

        if measure_flags[5]:
            next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)

            relative_waypoints = np.zeros((15, 2))
            for i, w_location in enumerate(waypoints):
                relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
            
            if len(waypoints) < 15:
                start_index = len(waypoints)
                # 【修復】：如果只剩 1 個點，強行去抓 start_index-2 會抓到陣列尾端都是 0 的垃圾值，導致座標大崩潰
                if start_index >= 2:
                    reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
                elif start_index == 1:
                    # 只剩一個點時，假設它跟車子的連線方向就是前進方向
                    reference_vector = relative_waypoints[0]
                else:
                    # 沒點了，往前填充 0 (已經被初始化為 0，這裡給極小值或沿用)
                    reference_vector = np.array([1.0, 0.0])
                    
                for i in range(start_index, 15):
                    relative_waypoints[i] = relative_waypoints[i-1] + reference_vector

            encoded_state['waypoints'] = relative_waypoints / 10.0 # 縮放座標以穩定梯度

        if measure_flags[6]: encoded_state['rgb_camera'] = env.observation
        if measure_flags[7]: encoded_state['seg_camera'] = env.observation
        if measure_flags[8]:
            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_vector'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]
        if measure_flags[9]:
            vehicle_location = vector(env.start_wp.transform.location)
            theta = np.deg2rad(env.start_wp.transform.rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_fixed'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]
        if measure_flags[10]:
            encoded_state['distance_goal'] = [[len(env.route_waypoints) - env.current_waypoint_index]]

        return encoded_state

    def decode_vae_state(z):
        with torch.no_grad():
            sample = torch.tensor(z)
            sample = vae.decode(sample).cpu()
            generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0)) * 255
        return generated_image
    if not vae:
        decode_vae_state = None
    return create_observation_space(), encode_state, decode_vae_state
