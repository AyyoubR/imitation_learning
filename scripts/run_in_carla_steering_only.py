"""Run a steering-only BC checkpoint in CARLA with a P-controller for speed.

The trained model predicts ONLY steering. Throttle and brake come from a
simple proportional controller that tracks a target cruise speed.

Run
---
    python scripts/run_in_carla_steering_only.py \\
        --checkpoint runs/bc_pilotnet_v1/checkpoints/best.pt \\
        --target-speed 30 --town Town04

Camera defaults match the training config (400x300, FOV=90, x=1.5 z=1.6).
"""
from __future__ import annotations

import argparse
import queue
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import carla
except ImportError:
    print("ERROR: the `carla` Python module is not on sys.path.", file=sys.stderr)
    raise

from inference import BCController  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to BC checkpoint (.pt)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town04")
    p.add_argument("--fps", type=int, default=20, help="Must match training fps")
    p.add_argument("--vehicle", default="vehicle.tesla.model3")
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    # Speed controller.
    p.add_argument("--target-speed", type=float, default=30.0,
                   help="Cruise target in km/h.")
    p.add_argument("--kp", type=float, default=0.1,
                   help="P gain on speed error (km/h -> throttle/brake).")
    p.add_argument("--creep-throttle", type=float, default=0.5,
                   help="Minimum throttle while below 2 km/h to avoid stall.")
    p.add_argument("--max-steer", type=float, default=0.7,
                   help="Safety clip on steering magnitude.")

    # Physics wake-up: CARLA vehicles spawned in sync mode often sit in
    # gear=0 with frozen wheels until control has been applied for a moment.
    p.add_argument("--bootstrap-s", type=float, default=1.5,
                   help="Seconds of fixed throttle before handing over to BC.")
    p.add_argument("--bootstrap-throttle", type=float, default=0.7)
    p.add_argument("--spawn-z-offset", type=float, default=0.3,
                   help="Lift spawn above ground to avoid mesh embedding.")

    # Camera — must match the training dataset.
    p.add_argument("--cam-w", type=int, default=400)
    p.add_argument("--cam-h", type=int, default=300)
    p.add_argument("--cam-fov", type=float, default=90.0)
    p.add_argument("--cam-x", type=float, default=1.5)
    p.add_argument("--cam-y", type=float, default=0.0)
    p.add_argument("--cam-z", type=float, default=1.6)

    # Optional highway-only filter for Town04.
    p.add_argument("--highway-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-lanes", type=int, default=3)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def carla_img_to_rgb(image: "carla.Image") -> np.ndarray:
    """CARLA returns BGRA uint8; the model expects HxWx3 RGB uint8."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()


def speed_kmh(vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5


def p_speed_controller(target_kmh, current_kmh, kp, creep):
    error = target_kmh - current_kmh

    throttle = np.clip(kp * error, 0.0, 1.0)

    # NO BRAKE until vehicle is already moving
    brake = 0.0

    # anti-stall
    if current_kmh < 2.0:
        throttle = max(throttle, creep)

    return float(throttle), 0.0


def count_parallel_driving_lanes(wp: "carla.Waypoint") -> int:
    if wp is None or wp.lane_type != carla.LaneType.Driving:
        return 0
    count = 1
    cursor = wp.get_right_lane()
    while cursor is not None and cursor.lane_type == carla.LaneType.Driving:
        count += 1
        cursor = cursor.get_right_lane()
    cursor = wp.get_left_lane()
    while (cursor is not None
           and cursor.lane_type == carla.LaneType.Driving
           and (cursor.lane_id * wp.lane_id) > 0):
        count += 1
        cursor = cursor.get_left_lane()
    return count


def filter_highway_spawns(world, spawn_points, min_lanes: int) -> list:
    carla_map = world.get_map()
    kept = []
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is not None and count_parallel_driving_lanes(wp) >= min_lanes:
            kept.append(sp)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    ctrl = BCController(args.checkpoint, device=args.device)
    ctrl.reset()  # clear temporal frame buffer (no-op for non-LSTM archs)
    print(f"Loaded BC controller from {args.checkpoint} (arch={ctrl.arch})")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    if world.get_map().name.split("/")[-1] != args.town:
        world = client.load_world(args.town)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    actors_to_destroy: list = []
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=4)
    camera = None

    def cleanup():
        print("\nCleaning up...")
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass
        for a in actors_to_destroy:
            try:
                a.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))

    try:
        bp_lib = world.get_blueprint_library()

        # --- Spawn vehicle ---
        vehicle_bp = bp_lib.filter(args.vehicle)[0]
        vehicle_bp.set_attribute("role_name", "hero")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in this town.")

        if args.highway_only:
            highway = filter_highway_spawns(world, spawn_points, args.min_lanes)
            if not highway:
                raise RuntimeError(
                    f"No highway spawns in {args.town} (min_lanes={args.min_lanes}). "
                    f"Pass --no-highway-only to disable the filter."
                )
            print(f"Highway filter: {len(highway)}/{len(spawn_points)} spawns kept.")
            spawn_points = highway

        spawn = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        
        actors_to_destroy.append(vehicle)
        print(f"Spawned {args.vehicle} at {spawn.location}")
        vehicle.set_simulate_physics(False)
        world.tick()
        vehicle.set_simulate_physics(True)
        vehicle.set_autopilot(False)
        
        for _ in range(5):
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, brake=0.0))
            world.tick()
        
        # --- Camera sensor ---
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.cam_w))
        cam_bp.set_attribute("image_size_y", str(args.cam_h))
        cam_bp.set_attribute("fov", str(args.cam_fov))
        cam_bp.set_attribute("sensor_tick", "0.0")
        cam_transform = carla.Transform(
            carla.Location(x=args.cam_x, y=args.cam_y, z=args.cam_z),
            carla.Rotation(),
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actors_to_destroy.append(camera)
        spectator = world.get_spectator()
        transform = vehicle.get_transform()

        spectator.set_transform(
            carla.Transform(
                transform.location + carla.Location(z=20),
                carla.Rotation(pitch=-90)
            )
        )
        def on_image(img):
            if image_queue.full():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    pass
            image_queue.put(img)

        camera.listen(on_image)

        # --- Collision sensor (for logging only) ---
        collision_bp = bp_lib.find("sensor.other.collision")
        collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors_to_destroy.append(collision)
        collisions = {"count": 0}
        collision.listen(lambda _e: collisions.update(count=collisions["count"] + 1))

        print("Location:", vehicle.get_location())
        # --- Settle physics ---
        for _ in range(3):
            world.tick()
        while not image_queue.empty():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                break

        # --- Control loop ---
        print(f"Driving for {args.duration_s:.0f}s at target={args.target_speed:.0f} km/h")
        step = 0
        t_end = time.time() + args.duration_s
        while time.time() < t_end:
            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                break
            print("Location:", vehicle.get_location())
            transform = vehicle.get_transform()

            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=20),
                    carla.Rotation(pitch=-90)
                )
            )
            rgb = carla_img_to_rgb(image)

            # steering model
            steer_raw, _, _ = ctrl.act(rgb)
            steer = float(np.clip(steer_raw, -args.max_steer, args.max_steer))

            v_kmh = speed_kmh(vehicle)

            # bootstrap phase (ONLY throttle override, no continue)
            if step < 10:
                throttle = 0.8
                brake = 0.0
            else:
                throttle, brake = p_speed_controller(
                    args.target_speed, v_kmh, args.kp, args.creep_throttle
                )

            vehicle.apply_control(carla.VehicleControl(
                steer=steer,
                throttle=throttle,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            ))
            if step % args.fps == 0:  # once per simulated second
                print(f"step={step:5d} v={v_kmh:5.1f} km/h  "
                      f"steer={steer:+.3f} thr={throttle:.2f} brk={brake:.2f}  "
                      f"coll={collisions['count']}")
            step += 1

    finally:
        cleanup()


if __name__ == "__main__":
    main()
