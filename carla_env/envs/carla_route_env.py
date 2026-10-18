import os
import subprocess
import sys
import glob
import time
import gym
import pygame
import cv2
from pygame.locals import *

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from collections import deque
import itertools

intersection_routes = itertools.cycle(
    [(57, 81), (70, 11), (70, 12), (78, 68), (74, 41), (42, 73), (71, 62), (74, 40), (71, 77), (6, 12), (65, 52), (63, 80)])
eval_routes = itertools.cycle([(48, 21), (0, 72), (28, 83), (61, 39)])

discrete_actions = {
    0: [-1, 1, 0], 1: [0, 1, 0], 2: [1, 1, 0], 3: [0, 0, 1],
}


class CarlaRouteEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1120, 560), obs_res=(160, 80),
                 reward_fn=None,
                 observation_space=None,
                 encode_state_fn=None, decode_vae_fn=None,
                 fps=15, action_smoothing=0.0, action_space_type="continuous",
                 activate_spectator=True,
                 activate_lidar=False,
                 start_carla=True,
                 eval=False,
                 activate_render=True):
        """
        A gym-like environment for interacting with a running CARLA environment and controlling a Lincoln MKZ2017 vehicle.

        Parameters:
            - host (str): IP address of the CARLA host
            - port (int): Port used to connect to CARLA
            - viewer_res (tuple[int, int]): Resolution of the spectator camera as a (width, height) tuple
            - obs_res (tuple[int, int]): Resolution of the observation camera as a (width, height) tuple
            - reward_fn (function): Custom reward function that is called every step. If None, no reward function is used.
            - observation_space: Custom observation space. If None, the default observation space is used.
            - encode_state_fn (function): Function that encodes the image from the observation camera to a state vector returned by step(). If None, the full image is returned.
            - decode_vae_fn (function): Function that decodes a state vector to an image. Used only if encode_state_fn is not None.
            - fps (int): FPS of the client. If fps <= 0 then use unbounded FPS.
            - action_smoothing (float): Scalar used to smooth the incoming action signal. 1.0 = max smoothing, 0.0 = no smoothing
            - action_space_type (str): Type of action space. Can be "continuous" or "discrete".
            - activate_spectator (bool): Whether to activate the spectator camera. Default is True.
            - activate_lidar (bool): Whether to activate the lidar sensor. Default is False.
            - start_carla (bool): Whether to automatically start CARLA when True. Note that you need to set the environment variable ${CARLA_ROOT} to point to the CARLA root directory for this option to work.
            - eval (bool): Whether the environment is used for evaluation or training. Default is False.
            - activate_render (bool): Whether to activate rendering. Default is True.
        """

        self.carla_process = None
        if start_carla:
            if "CARLA_ROOT" not in os.environ:
                raise Exception("${CARLA_ROOT} has not been set!")
            carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            launch_command += [f'-carla-world-port={port}']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("Waiting for CARLA to initialize\n")

            # ./CarlaUE4.sh -quality_level=Low -benchmark -fps=15 -RenderOffScreen
            time.sleep(5)

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.activate_render = activate_render


        # Setup gym environment
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            # 支援 3D 動作空間: [steer, throttle, brake]
            self.action_space = gym.spaces.Box(np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        elif self.action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(len(discrete_actions))

        self.observation_space = observation_space

        self.fps = fps
        self.action_smoothing = action_smoothing
        self.episode_idx = -2

        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.decode_vae_fn = None if not callable(decode_vae_fn) else decode_vae_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000  # m
        self.activate_spectator = activate_spectator
        self.activate_lidar = activate_lidar
        self.eval = eval

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings
            
            # self.world.set_weather(carla.WeatherParameters.MidRainyNoon)


            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud and initialize pygame for visualization
            if self.activate_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            seg_settings = {}
            if "seg_camera" in self.observation_space.keys():
                seg_settings.update({
                    'camera_type': "sensor.camera.semantic_segmentation",
                    'custom_palette': True
                })
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=sensor_transforms["dashboard"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  **seg_settings)

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            if self.activate_lidar:
                self.lidar = Lidar(self.world, transform=sensor_transforms["lidar"],
                                   attach_to=self.vehicle, on_recv_image=lambda e: self._set_lidar_data(e))
        except Exception as e:
            self.close()
            raise e
        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
        return [seed]

    def reset(self, is_training=False):
        # Create new route
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        # Two different variables to differ between success episode and fail episode
        self.terminal_state = False  # Set to True when we want to end episode
        self.success_state = False  # Set to True when we want to end episode.

        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.lidar_data = self.lidar_data_buffer = None
        self.step_count = 0

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.low_speed_timer = 0.0
        self.signed_distance_from_center = 0.0
        self.route_lookahead_angle = 0.0
        self.collision_detected = False
        self.collision_actor = ""
        self.collision_actor_type = ""
        self.start_time = time.time()
        
        # 修正: 重置紅燈相關狀態，避免跨 Episode 污染導致剛重置就立刻判定闖紅燈
        self.prev_is_at_red_light = False
        self.last_traffic_light = None
        self.is_at_red_light = False
        self.violated_red_light = False
        self.red_light_grace_timer = 0.0
        self._was_at_red_light_for_smoothing = False
        # 綠燈重新啟動期間，清除紅燈強制煞車留下的殘留值。
        self._green_release_steps = 0
        self._red_hold_steer = 0.0
        self._holding_red_steer = False
        self.route_curve_angle = 0.0
        self._curve_target_speed = 26.0
        self._reward_prev_steer = 0.0
        self.progress_since_last_step = 0.0
        
        self.world.tick()
        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)
        return obs

    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.set_simulate_physics(False)  # Reset the car's physics
        
        # 重置平滑動作狀態
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_pedal = 0.0
        self._green_release_steps = 0
        self._red_hold_steer = 0.0
        self._holding_red_steer = False
        self.route_curve_angle = 0.0
        self._curve_target_speed = 26.0
        self._reward_prev_steer = 0.0
        self.progress_since_last_step = 0.0

        # 路線切換時同步清除上一段路線的紅燈與違規狀態，避免新路線
        # 繼承強制煞車或被誤判為仍在紅燈區。
        self.prev_is_at_red_light = False
        self.last_traffic_light = None
        self.is_at_red_light = False
        self.violated_red_light = False
        self.red_light_grace_timer = 0.0
        self._was_at_red_light_for_smoothing = False

        # Clear control state left by the previous route. In particular, a
        # stale brake must not survive when physics is enabled for the new one.
        self.vehicle.control.brake = float(0.0)
        self.vehicle.control.hand_brake = False
        self.vehicle.control.manual_gear_shift = False

        # Generate waypoints along the lap
        if not self.eval:
            if self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
                spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(intersection_routes)]
            else:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        else:
            maneuver = "INVALID"

        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s" % maneuver,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.activate_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
            # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos_observation)

        pos_vae_decoded = (self.display.get_size()[0] - 2 * obs_w - 10, 10)
        if self.decode_vae_fn:
            self.display.blit(pygame.surfarray.make_surface(self.observation_decoded.swapaxes(0, 1)), pos_vae_decoded)

        if self.activate_lidar:
            lidar_h, lidar_w = self.lidar_data.shape[:2]
            pos_lidar = (self.display.get_size()[0] - obs_w - 10, 100)
            self.display.blit(pygame.surfarray.make_surface(self.lidar_data.swapaxes(0, 1)), pos_lidar)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()



    def _find_route_traffic_light(self, max_lookahead=35.0):
        """Return the closest traffic light whose trigger volume intersects the route ahead."""
        if not hasattr(self, "route_waypoints") or not self.route_waypoints:
            return None, 999.0

        start_index = max(0, int(getattr(self, "current_waypoint_index", 0)))
        stop_index = min(len(self.route_waypoints),
                         start_index + int(max_lookahead) + 8)
        route_slice = self.route_waypoints[start_index:stop_index]
        vehicle_location = vector(self.vehicle.actor.get_transform().location)

        best_light = None
        best_distance = 999.0
        for traffic_light in self.world.get_actors().filter("*traffic_light*"):
            trigger = traffic_light.trigger_volume
            trigger_location = carla.Location(
                x=trigger.location.x, y=trigger.location.y, z=trigger.location.z
            )
            traffic_light.get_transform().transform(trigger_location)
            trigger_vector = vector(trigger_location)

            # Trigger volumes span the affected lanes. Matching them to the
            # planned route avoids stopping for perpendicular/adjacent lights.
            route_threshold = max(5.0, float(trigger.extent.x) + 2.0,
                                  float(trigger.extent.y) + 2.0)
            affects_route = any(
                np.linalg.norm(
                    vector(waypoint.transform.location)[:2] - trigger_vector[:2]
                ) <= route_threshold
                for waypoint, _ in route_slice
            )
            if not affects_route:
                continue

            distance = float(np.linalg.norm(
                vehicle_location[:2] - trigger_vector[:2]
            ))
            if distance <= max_lookahead and distance < best_distance:
                best_light = traffic_light
                best_distance = distance

        return best_light, best_distance

    def _is_red_light_violation(self):
        """Detect the relevant light early enough to stop and flag a true crossing."""
        was_at_red_light = bool(getattr(self, "prev_is_at_red_light", False))
        previous_light = getattr(self, "last_traffic_light", None)

        is_in_native_trigger = self.vehicle.actor.is_at_traffic_light()
        native_light = (self.vehicle.actor.get_traffic_light()
                        if is_in_native_trigger else None)
        route_light, route_distance = self._find_route_traffic_light()
        traffic_light = native_light or route_light

        if traffic_light is not None:
            self.last_traffic_light = traffic_light

            # During training, expose stop/restart transitions without forcing
            # every intersection to be red.
            if not self.eval and getattr(self, "last_forced_light_id", None) != traffic_light.id:
                if (traffic_light.get_state() != carla.TrafficLightState.Red
                        and np.random.rand() < 0.3):
                    traffic_light.set_state(carla.TrafficLightState.Red)
                    traffic_light.set_red_time(4.0 + np.random.rand() * 4.0)
                self.last_forced_light_id = traffic_light.id

        state = (traffic_light.get_state()
                 if traffic_light is not None else None)
        is_at_red_light = state in (
            carla.TrafficLightState.Red,
            carla.TrafficLightState.Yellow,
        )

        if native_light is not None:
            native_location = native_light.get_transform().location
            distance_to_light = self.vehicle.actor.get_location().distance(native_location)
        else:
            distance_to_light = route_distance

        dt = 1.0 / max(float(self.fps), 1.0)
        if is_at_red_light:
            self.red_light_grace_timer = 2.0
        else:
            previous_state = (previous_light.get_state()
                              if previous_light is not None else None)
            if (previous_state == carla.TrafficLightState.Red
                    and getattr(self, "red_light_grace_timer", 0.0) > 0.0):
                # CARLA can drop the native trigger for a frame near the stop
                # line. Preserve the state briefly; if the car truly crosses,
                # the expiry below produces the violation.
                self.red_light_grace_timer = max(
                    0.0, self.red_light_grace_timer - dt
                )
                is_at_red_light = self.red_light_grace_timer > 0.0
                traffic_light = previous_light
                trigger = previous_light.trigger_volume
                trigger_location = carla.Location(
                    x=trigger.location.x, y=trigger.location.y, z=trigger.location.z
                )
                previous_light.get_transform().transform(trigger_location)
                distance_to_light = self.vehicle.actor.get_location().distance(
                    trigger_location
                )
            else:
                self.red_light_grace_timer = 0.0

        violation = False
        if was_at_red_light and not is_at_red_light:
            violation_light = previous_light or traffic_light
            if (violation_light is not None
                    and violation_light.get_state() == carla.TrafficLightState.Red
                    and self.vehicle.get_speed() > 2.0):
                violation = True

        if not is_at_red_light and traffic_light is None:
            self.last_traffic_light = None

        self.distance_to_red_light = float(distance_to_light)
        self.prev_is_at_red_light = is_at_red_light
        return is_at_red_light, violation

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # 【修復】紅燈偵測移到 world.tick() 之後，見下方
        # (不再在此處提前偵測，避免用到 stale 狀態判定闖紅燈)

        # Take action
        if action is not None:
            brake_action = 0.0
            throttle = 0.0
            steer = 0.0

            if self.action_space_type == "continuous":
                if len(action) == 2:
                    steer, pedal = [float(a) for a in action]
                    # The legacy 2D action already uses a signed pedal.
                    pedal = max(-1.0, min(1.0, pedal))
                else:
                    # 3D output: [steer, throttle, brake]
                    steer = float(action[0])
                    raw_throttle = float(action[1])
                    raw_brake = float(action[2])
                    raw_throttle = max(0.0, min(1.0, raw_throttle))
                    raw_brake = max(0.0, min(1.0, raw_brake))
                    # Combine both pedals into one signed value. Give throttle
                    # priority on a tie so initial [0.5, 0.5] exploration still
                    # moves the car; brake wins only when it is strictly larger.
                    pedal = (raw_throttle if raw_throttle >= raw_brake
                             else -raw_brake)
            elif self.action_space_type == "discrete":
                steer, raw_throttle, raw_brake = discrete_actions[action]
                pedal = (raw_throttle if raw_throttle >= raw_brake
                         else -raw_brake)

            # Clip actions just in case
            steer = max(-1.0, min(1.0, steer))
            pedal = max(-1.0, min(1.0, pedal))

            red_light = bool(getattr(self, 'is_at_red_light', False))
            was_at_red_light = getattr(self, '_was_at_red_light_for_smoothing', False)
            released_red_light = was_at_red_light and not red_light
            alpha = max(0.0, min(1.0, float(self.action_smoothing)))

            # Keep following the route while approaching a red light. Freeze
            # steering only after the vehicle is actually stationary so the
            # policy cannot wind the wheels during the wait.
            if red_light and self.vehicle.get_speed() < 1.0:
                self._holding_red_steer = True
                steer_target = 0.0
            else:
                self._holding_red_steer = False
                # Pure-pursuit lookahead starts turning before the current
                # waypoint orientation changes. This is essential on Town02's
                # short, sharp intersection curves.
                vehicle_speed = self.vehicle.get_speed()
                previous_curve_angle = abs(float(
                    getattr(self, 'route_curve_angle', 0.0)
                ))
                if previous_curve_angle > 0.35:
                    # A long chord cuts inside Town02's tight corners. Once
                    # the route preview sees a real bend, track a nearer point
                    # until the curve is cleared.
                    lookahead_steps = int(round(3.0 + 0.08 * vehicle_speed))
                else:
                    lookahead_steps = int(round(5.0 + 0.20 * vehicle_speed))
                lookahead_steps = max(3, lookahead_steps)
                target_index = min(
                    len(self.route_waypoints) - 1,
                    self.current_waypoint_index + lookahead_steps,
                )
                target_location = vector(
                    self.route_waypoints[target_index][0].transform.location
                )
                vehicle_location = vector(self.vehicle.get_location())
                target_vector = target_location - vehicle_location
                forward_vector = vector(
                    self.vehicle.get_transform().rotation.get_forward_vector()
                )
                lookahead_angle = angle_diff(forward_vector, target_vector)
                self.route_lookahead_angle = float(lookahead_angle)

                # Use a separate, longer preview for speed planning. A single
                # pure-pursuit target can briefly line up with the vehicle in
                # the middle of a turn, even though the route is still curving.
                # That previously released the brake and let the policy
                # accelerate again before the steering had caught up.
                route_forward = vector(
                    self.current_waypoint.transform.get_forward_vector()
                )
                preview_steps = int(round(
                    12.0 + 0.35 * self.vehicle.get_speed()
                ))
                preview_stop = min(
                    len(self.route_waypoints) - 1,
                    self.current_waypoint_index + preview_steps,
                )
                preview_angle = abs(float(lookahead_angle))
                for preview_index in range(
                        self.current_waypoint_index + 2,
                        preview_stop + 1,
                        2):
                    preview_forward = vector(
                        self.route_waypoints[preview_index][0]
                        .transform.get_forward_vector()
                    )
                    preview_angle = max(
                        preview_angle,
                        abs(float(angle_diff(route_forward, preview_forward))),
                    )
                self.route_curve_angle = float(preview_angle)

                # CARLA world coordinates are left-handed, hence the
                # cross-track correction sign below.
                cross_track_error = float(
                    getattr(self, 'signed_distance_from_center', 0.0)
                )
                # The lookahead vector already contains cross-track error.
                # The previous 1.10/-0.15 gains corrected that error twice;
                # with EMA and the slew limiter this produced an expanding
                # left-right limit cycle on long straights. Keep a small
                # explicit centering term, but let pure pursuit dominate.
                route_correction = (
                    0.75 * lookahead_angle - 0.05 * cross_track_error
                )
                steer_target = 0.25 * steer + route_correction

            # EMA alone still allowed the trained policy to reverse steering
            # by ~0.1 every 1/15 s. Apply a speed-aware slew-rate limit after
            # smoothing: 0.6 control unit/s at rest, 0.3 unit/s at 30 km/h.
            previous_steer = getattr(self, 'prev_steer', 0.0)
            urgent_reversal = (
                abs(float(getattr(self, 'route_lookahead_angle', 0.0))) >= 0.20
                and steer_target * previous_steer < 0.0
            )
            steering_alpha = min(alpha, 0.25) if urgent_reversal else alpha
            smoothed_steer = (steering_alpha * previous_steer
                              + (1.0 - steering_alpha) * steer_target)
            vehicle_speed = self.vehicle.get_speed()
            speed_ratio = max(0.0, min(1.0, vehicle_speed / 30.0))
            max_steer_magnitude = max(0.25, 0.70 - 0.015 * vehicle_speed)
            smoothed_steer = max(-max_steer_magnitude,
                                 min(max_steer_magnitude, smoothed_steer))
            steer_slew_rate = 0.6 - 0.3 * speed_ratio
            if abs(float(getattr(self, 'route_lookahead_angle', 0.0))) > 0.25:
                # A global high slew rate recreates straight-line wobble. Only
                # relax it while a real turn is in progress, where delayed
                # counter-steer caused the acceptance-test curb impacts.
                steer_slew_rate = max(
                    steer_slew_rate, 0.9 - 0.3 * speed_ratio
                )
                if urgent_reversal:
                    steer_slew_rate = max(steer_slew_rate, 1.0)
            max_steer_delta = steer_slew_rate / max(float(self.fps), 1.0)
            steer_delta = max(-max_steer_delta,
                              min(max_steer_delta, smoothed_steer - previous_steer))
            steer = previous_steer + steer_delta
            self.prev_steer = steer

            if red_light:
                # Approach a distant red light instead of braking at the first
                # 35 m detection. The target falls linearly to zero near the
                # trigger volume; full braking is reserved for the stop zone.
                distance_to_light = max(
                    0.0, float(getattr(self, 'distance_to_red_light', 0.0))
                )
                current_speed = self.vehicle.get_speed()
                target_approach_speed = max(
                    0.0, min(25.0, 1.5 * (distance_to_light - 5.0))
                )
                if distance_to_light <= 6.0:
                    pedal = -1.0
                elif current_speed > target_approach_speed + 2.0:
                    speed_excess = current_speed - target_approach_speed
                    pedal = -min(1.0, 0.25 + speed_excess / 15.0)
                else:
                    # The old policy maps any non-zero red-light observation
                    # to brake. Override that only while there is still enough
                    # distance to approach the stop line safely.
                    pedal = min(0.45, max(0.30, pedal))

                self.prev_pedal = pedal
                self.prev_throttle = max(0.0, pedal)
                self.prev_brake = max(0.0, -pedal)
            else:
                # On the first green frame, release the forced brake first,
                # then ramp the learned pedal command instead of bypassing
                # smoothing and causing an acceleration/braking shock.
                if released_red_light:
                    # Discard the forced -1 brake immediately. Keep restart
                    # assistance active until the car is moving, not merely
                    # for a handful of frames.
                    self.prev_pedal = 0.0
                    self._green_release_steps = max(1, int(round(self.fps * 3.0)))

                if self._green_release_steps > 0:
                    current_speed = self.vehicle.get_speed()
                    if current_speed >= 5.0:
                        self._green_release_steps = 0
                    else:
                        # 0.12 was below the useful launch range of the Model
                        # 3 after EMA filtering. 0.35 reliably releases the
                        # car without producing a high-speed acceleration jump.
                        pedal = max(0.35, pedal)
                        self.prev_pedal = (
                            alpha * max(0.0, getattr(self, 'prev_pedal', 0.0))
                            + (1.0 - alpha) * pedal
                        )
                        self._green_release_steps -= 1

                if self._green_release_steps <= 0:
                    self.prev_pedal = (alpha * getattr(self, 'prev_pedal', 0.0)
                                       + (1.0 - alpha) * pedal)
                pedal = self.prev_pedal
                self.prev_throttle = max(0.0, pedal)
                self.prev_brake = max(0.0, -pedal)

            # Slow down before a curve rather than relying on a steering clamp
            # to save a 25-30 km/h entry. Apply this after both traffic-light
            # branches: a red-light approach must not disable curve safety.
            curve_angle = max(
                abs(float(getattr(self, 'route_lookahead_angle', 0.0))),
                abs(float(getattr(self, 'route_curve_angle', 0.0))),
                0.8 * abs(float(previous_steer)),
            )
            raw_curve_target = max(
                8.0, min(26.0, 26.0 - 40.0 * curve_angle)
            )
            previous_curve_target = float(
                getattr(self, '_curve_target_speed', 26.0)
            )
            if raw_curve_target <= previous_curve_target:
                curve_target_speed = raw_curve_target
            else:
                # Drop the target immediately on curve entry. Restore it only
                # after steering and cross-track error have both settled; an
                # angle zero-crossing alone does not mean the turn is over.
                curve_settled = (
                    abs(float(getattr(self, 'route_lookahead_angle', 0.0))) < 0.08
                    and abs(float(previous_steer)) < 0.18
                    and abs(float(getattr(
                        self, 'signed_distance_from_center', 0.0
                    ))) < 0.5
                )
                if curve_settled:
                    curve_target_speed = min(
                        raw_curve_target,
                        previous_curve_target
                        + 4.0 / max(float(self.fps), 1.0),
                    )
                else:
                    curve_target_speed = previous_curve_target
            self._curve_target_speed = float(curve_target_speed)

            current_speed = self.vehicle.get_speed()
            if current_speed > curve_target_speed:
                speed_excess = current_speed - curve_target_speed
                curve_brake = min(0.80, 0.15 + speed_excess / 16.0)
                pedal = min(pedal, -curve_brake)
                self.prev_pedal = pedal
                self.prev_throttle = 0.0
                self.prev_brake = curve_brake
            elif curve_target_speed < 25.0 and pedal > 0.0:
                # Do not let a high policy action overshoot the held target
                # while the vehicle is still laterally loaded in the turn.
                curve_throttle_limit = min(
                    0.35,
                    max(0.0, 0.12 * (curve_target_speed - current_speed)),
                )
                pedal = min(pedal, curve_throttle_limit)
                self.prev_pedal = pedal
                self.prev_throttle = max(0.0, pedal)
                self.prev_brake = max(0.0, -pedal)

            # A poor policy can also stop after ordinary curve braking.
            # Recover after 0.5 s of low speed on green, using the same launch
            # floor that proved reliable after traffic lights.
            if (not red_light
                    and current_speed < 1.0
                    and getattr(self, 'low_speed_timer', 0.0) >= 0.5):
                pedal = max(0.35, pedal)
                self.prev_pedal = pedal
                self.prev_throttle = pedal
                self.prev_brake = 0.0

            self._was_at_red_light_for_smoothing = red_light
            throttle = max(0.0, min(1.0, pedal))
            brake_action = max(0.0, min(1.0, -pedal))

            # Limit throttle at high speed, but never weaken a red-light stop.
            if not red_light and self.vehicle.get_speed() > 30.0:
                throttle = min(throttle, 0.5)

            # 強制將控制指令應用到車輛
            self.vehicle.control.steer = float(steer)
            self.vehicle.control.throttle = float(throttle)
            self.vehicle.control.brake = float(brake_action)
            self.vehicle.control.hand_brake = False
            # 【修復】移除 gear=1 鎖定，讓 CARLA 自動變速箱正常運作
            # 固定 1 檔會導致高速時引擎扭力不足，車輛無法維持目標速度
            self.vehicle.control.manual_gear_shift = False
            
            # 確保指令被送出給 Carla
            self.vehicle.actor.apply_control(self.vehicle.control)
            # 設置標記，阻止 Vehicle.tick() 中的 apply_control 覆蓋我們的精確值
            self.vehicle._control_applied_this_tick = True
                
        # Tick game
        self.world.tick()
        
        # 【修復】紅燈偵測移到 world.tick() 之後
        # 這樣判定的是「這一步物理執行完畢後」的真實狀態，
        # 而非上一步的殘留值（避免 AI 正確煞車卻因舊位置被誤判闖紅燈）
        self.is_at_red_light, self.violated_red_light = self._is_red_light_violation()
        
        if self.violated_red_light:
            self.terminal_state = True  # 觸發終止

        # Check for new route completion after environment has ticked and position updated
        if self.current_waypoint_index >= len(self.route_waypoints) - 1:
            if not self.eval:
                self.new_route()
                # Important: after creating a new route, we must tick the world again
                # and update the actor states properly, otherwise we feed stale/jumping observations
                self.world.tick()
            else:
                self.success_state = True

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image()

        if self.activate_lidar:
            self.lidar_data = self._get_lidar_data()

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        
        # 【修復：防止停等紅燈時的 Waypoint 閃爍跳躍】
        # 當車速極慢 (例如停等紅燈) 時，因物理引擎微小震動，dot product 判定會在 >0 和 <0 間跳躍，
        # 導致 AI 看到的「前方路徑」瘋狂閃爍、或判定錯亂而錯過轉彎點。
        # 解法：如果車速極低且沒有實際前進，我們凍結 waypoint_index，不進行跳號判定。
        speed = self.vehicle.get_speed()
        if speed > 0.5:
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                if next_waypoint_index >= len(self.route_waypoints):
                    break
                wp, _ = self.route_waypoints[next_waypoint_index]
                
                # 計算是否越過該節點。加入 0.1 緩衝區避免浮點數震盪 (Hysteresis)
                dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                             vector(transform.location - wp.transform.location)[:2])
                if dot > 0.1:  # Did we truly pass the waypoint?
                    waypoint_index += 1  # Go to next waypoint
                else:
                    break
            self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[self.current_waypoint_index + 1]
        else:
            # 【修復】：如果已經走到最後一個導航點，強行讓 next_waypoint = 目前的點
            # 這樣才不會用到上一個路線殘提出來的 stale (過期) waypoint 導致偏差計算大爆炸！
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[-1]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[self.current_waypoint_index]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate signed and absolute deviation from the route segment.
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            route_start = vector(self.current_waypoint.transform.location)
            route_end = vector(self.next_waypoint.transform.location)
            vehicle_position = vector(transform.location)
            route_direction = route_end[:2] - route_start[:2]
            vehicle_offset = vehicle_position[:2] - route_start[:2]
            route_norm = np.linalg.norm(route_direction)
            if route_norm > 1e-6:
                self.signed_distance_from_center = float(
                    (route_direction[0] * vehicle_offset[1]
                     - route_direction[1] * vehicle_offset[0]) / route_norm
                )
                self.distance_from_center = abs(self.signed_distance_from_center)
            else:
                self.signed_distance_from_center = 0.0
                self.distance_from_center = float(np.linalg.norm(vehicle_offset))
        else:
            # No center offset on the very last waypoint to prevent explosion.
            self.signed_distance_from_center = 0.0
            self.distance_from_center = 0.0
        
        self.center_lane_deviation += self.distance_from_center

        # Calculate per-step route progress before updating the cumulative
        # distance. The reward uses this to distinguish moving forward from
        # collecting reward while idling.
        self.progress_since_last_step = 0.0
        if action is not None:
            self.progress_since_last_step = self.previous_location.distance(transform.location)
            self.distance_traveled += self.progress_since_last_step
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        # Encode the state
        encoded_state = self.encode_state_fn(self)
        if self.decode_vae_fn:
            self.observation_decoded = self.decode_vae_fn(encoded_state['vae_latent'])
        self.step_count += 1

        # DEBUG: Draw path
        # self._draw_path_server(life_time=1.0, skip=8)
        # DEBUG: Draw current waypoint
        # self.world.debug.draw_point(self.current_waypoint.transform.location + carla.Location(z=1.25), size=0.1,color=carla.Color(0, 255, 255), life_time=2.0, persistent_lines=False)

        # Check for ESC press
        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True
            self.render()

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }
        
        is_done = self.terminal_state or self.success_state
        if is_done:
            # 模擬 Monitor Wrapper 的行為，讓 SB3 能讀到 Tensorboard 指標
            info['episode'] = {
                'r': self.total_reward,
                'l': self.step_count,
                't': round(time.time() - getattr(self, 'start_time', time.time()), 6)
            }
            
        return encoded_state, self.last_reward, is_done, info

    def _draw_path_server(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints) - 1, skip + 1):
            z = 30.25
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i + 1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=z),
                w1.transform.location + carla.Location(z=z),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=z), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=z), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(camera.actor.attributes['image_size_x'])
        image_h = int(camera.actor.attributes['image_size_y'])
        fov = float(camera.actor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_lidar_data(self):
        while self.lidar_data_buffer is None:
            pass
        image = self.lidar_data_buffer.copy()
        self.lidar_data_buffer = None
        return image

    def _on_collision(self, event):
        actor_name = get_actor_display_name(event.other_actor)
        if actor_name != "Road":
            self.collision_actor = actor_name
            self.collision_actor_type = getattr(event.other_actor, 'type_id', '')
            self.collision_detected = True
            self.terminal_state = True
        if self.activate_render:
            self.hud.notification("Collision with {}".format(actor_name))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.activate_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_lidar_data(self, image):
        self.lidar_data_buffer = image
