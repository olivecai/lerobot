# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Start the client once and drive it interactively — no reconnect overhead between commands.

```shell
python src/lerobot/async_inference/robot_client.py \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \\
    --robot.id=black \\
    --server_address=127.0.0.1:8080 \\
    --policy_type=act \\
    --pretrained_name_or_path=user/model \\
    --policy_device=mps \\
    --client_device=cpu \\
    --actions_per_chunk=50 \\
    --chunk_size_threshold=0.5 \\
    --aggregate_fn_name=weighted_average
```

Available commands once the REPL is running:

  status           Capture and publish one observation to the policy server
                   (visible immediately via /status, /observation, /images).
                   Returns to the prompt without executing any policy.

  run <task>       Send policy instructions to the server and start the
                   control loop. <task> is the instruction string, e.g.
                       run fold the t-shirt

  stop             Stop the current policy rollout and return to the prompt.
                   The gRPC channel stays open; you can run another command
                   immediately.

  quit             Stop any active rollout, disconnect, and exit.
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.server_address = config.server_address

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        # Per-rollout state — reset between runs via _reset_rollout_state()
        self.shutdown_event = threading.Event()
        self.shutdown_event.set()  # not running until 'run' is issued

        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []

        # Barrier is recreated each run (2 threads: action receiver + control loop)
        self.start_barrier = threading.Barrier(2)

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.must_go = threading.Event()
        self.must_go.set()

        # Thread handles — populated by start_rollout(), cleared by stop_rollout()
        self._action_receiver_thread: threading.Thread | None = None
        self._control_loop_thread: threading.Thread | None = None

        self.logger.info("Robot connected and ready")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Ping the policy server to confirm the channel is reachable."""
        try:
            start = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            self.logger.debug(f"Connected to policy server in {time.perf_counter() - start:.4f}s")
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def disconnect(self):
        """Disconnect the robot and close the gRPC channel."""
        self.robot.disconnect()
        self.logger.debug("Robot disconnected")
        self.channel.close()
        self.logger.debug("gRPC channel closed")

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    def _reset_rollout_state(self):
        """Reset all per-rollout state so a fresh run can start cleanly."""
        self.shutdown_event.clear()

        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1

        with self.action_queue_lock:
            self.action_queue = Queue()
        self.action_queue_size = []

        self.start_barrier = threading.Barrier(2)
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.must_go = threading.Event()
        self.must_go.set()

    @property
    def rollout_active(self) -> bool:
        return not self.shutdown_event.is_set()

    def start_rollout(self, task: str) -> bool:
        """Send policy instructions and spawn the control-loop threads.

        Args:
            task: Task instruction string passed to the policy server.

        Returns:
            True if the rollout started successfully, False otherwise.
        """
        if self.rollout_active:
            self.logger.warning("A rollout is already active. Call stop_rollout() first.")
            return False

        self._reset_rollout_state()

        try:
            policy_config = RemotePolicyConfig(
                self.config.policy_type,
                self.config.pretrained_name_or_path,
                self.lerobot_features,
                self.config.actions_per_chunk,
                self.config.policy_device,
            )
            policy_config_bytes = pickle.dumps(policy_config)
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_config_bytes))
            self.logger.info(
                f"Policy instructions sent | type={policy_config.policy_type} | "
                f"model={policy_config.pretrained_name_or_path} | task='{task}'"
            )
        except grpc.RpcError as e:
            self.logger.error(f"Failed to send policy instructions: {e}")
            self.shutdown_event.set()
            return False

        self._action_receiver_thread = threading.Thread(
            target=self.receive_actions, daemon=True, name="action-receiver"
        )
        self._control_loop_thread = threading.Thread(
            target=self.control_loop, args=(task,), daemon=True, name="control-loop"
        )

        self._action_receiver_thread.start()
        self._control_loop_thread.start()

        self.logger.info(f"Rollout started for task: '{task}'")
        return True

    def stop_rollout(self):
        """Stop the active rollout and join its threads.

        The gRPC channel remains open so subsequent commands can reuse it.
        """
        if not self.rollout_active:
            self.logger.info("No active rollout to stop.")
            return

        self.logger.info("Stopping rollout...")
        self.shutdown_event.set()

        if self._control_loop_thread and self._control_loop_thread.is_alive():
            self._control_loop_thread.join(timeout=5)
        if self._action_receiver_thread and self._action_receiver_thread.is_alive():
            self._action_receiver_thread.join(timeout=5)

        self._control_loop_thread = None
        self._action_receiver_thread = None

        if self.config.debug_visualize_queue_size and self.action_queue_size:
            visualize_action_queue_size(self.action_queue_size)

        self.logger.info("Rollout stopped. Ready for next command.")

    # ------------------------------------------------------------------
    # Status-only observation
    # ------------------------------------------------------------------

    def publish_current_status(self) -> bool:
        """Capture and send a single observation without rolling out a policy.

        Sends through the normal SendObservations gRPC path so the policy
        server's FastAPI layer (/status, /observation, /images) reflects the
        current robot state immediately. must_go=True ensures the server stores
        it unconditionally.

        Returns True if successful, False otherwise.
        """
        self.logger.info("Capturing current robot state...")

        # Temporarily clear shutdown_event so send_observation() doesn't raise
        was_shut_down = self.shutdown_event.is_set()
        self.shutdown_event.clear()

        try:
            raw_observation: RawObservation = self.robot.get_observation()
            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=0,
            )
            observation.must_go = True

            success = self.send_observation(observation)
            if success:
                self.logger.info(
                    f"Status observation sent (timestep=0, must_go=True) | "
                    f"keys: {list(raw_observation.keys())}"
                )
            return success

        except Exception as e:
            self.logger.error(f"Error capturing/sending status observation: {e}")
            return False

        finally:
            # Restore shutdown state — if no rollout is active we want it set
            if was_shut_down:
                self.shutdown_event.set()

    # ------------------------------------------------------------------
    # Core send/receive
    # ------------------------------------------------------------------

    def send_observation(self, obs: TimedObservation) -> bool:
        """Send observation to the policy server."""
        if self.shutdown_event.is_set():
            raise RuntimeError("Client not running.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        self.logger.debug(f"Observation serialization time: {time.perf_counter() - start_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            self.logger.debug(f"Sent observation #{obs.get_timestep()}")
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            if new_action.get_timestep() <= latest_action:
                continue
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server (runs in its own thread)."""
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.rollout_active:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()

                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                if len(timed_actions) > 0:
                    self.logger.debug(
                        f"Received actions on device: {timed_actions[0].get_action().device.type}"
                    )

                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    self.logger.info(
                        f"Received action chunk for step #{timed_actions[0].get_timestep()} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency: {(receive_time - timed_actions[0].get_timestamp()) * 1000:.2f}ms | "
                        f"Deserialization: {deserialize_time * 1000:.2f}ms"
                    )

                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                self.must_go.set()

                if verbose:
                    new_size, new_timesteps = self._inspect_action_queue()
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )

            except grpc.RpcError as e:
                if self.rollout_active:
                    self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size} | "
                f"Pop took {get_end:.6f}s"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )
                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | "
                    f"Capture took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined control loop — runs in its own thread during a rollout."""
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.rollout_active:
            control_loop_start = time.perf_counter()

            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(
                f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}"
            )
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

REPL_HELP = """
Commands:
  status           Publish one observation to the policy server and return
                   to the prompt (no policy executed).
  run <task>       Start a policy rollout for the given task, e.g.:
                       run fold the t-shirt
  stop             Stop the current rollout and return to the prompt.
  quit             Stop any active rollout, disconnect, and exit.
  help             Show this message.
"""


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = RobotClient(cfg)

    if not client.connect():
        logging.error("Could not reach the policy server. Exiting.")
        return

    print(REPL_HELP)

    try:
        while True:
            try:
                raw = input("robot> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl-D or Ctrl-C — treat as quit
                raw = "quit"

            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "help":
                print(REPL_HELP)

            elif cmd == "status":
                if client.rollout_active:
                    print("A rollout is currently active. Run 'stop' first.")
                else:
                    client.publish_current_status()

            elif cmd == "run":
                if not arg:
                    print("Usage: run <task>  e.g.  run fold the t-shirt")
                    continue
                if client.rollout_active:
                    print("A rollout is already active. Run 'stop' first.")
                    continue
                client.start_rollout(task=arg)

            elif cmd == "stop":
                client.stop_rollout()

            elif cmd == "quit":
                client.stop_rollout()
                break

            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")

    finally:
        client.disconnect()
        logging.info("Client exited.")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()