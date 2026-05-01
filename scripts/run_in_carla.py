"""Run a trained BC checkpoint inside a live CARLA simulation.

Prerequisites
-------------
* A running CARLA server (e.g. ``./CarlaUE4.sh -quality-level=Low``).
* The ``carla`` Python API available (from CARLA's PythonAPI/carla/dist/*.egg).
* A checkpoint trained with this repo.

Run
---
    python scripts/run_in_carla.py \\
        --checkpoint runs/bc_pilotnet_v1/checkpoints/best.pt \\
        --host 127.0.0.1 --port 2000 --town Town04 --fps 20

Defaults to Town04 with spawning restricted to the highway ring (multi-lane
segments outside the inner city). Use ``--no-highway-only`` to allow any spawn
point, or tune ``--min-lanes`` to widen/narrow the highway filter.

The camera is configured to match the training data (400x300, FOV=90, mounted
at x=1.5 y=0 z=1.6). Change the CLI flags if your training config differs.
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

# Make the project package importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import carla
except ImportError as e:
    print("ERROR: the `carla` Python module is not on sys.path.\n"
          "Install the egg that ships with your CARLA build, e.g.:\n"
          "  easy_install CARLA_X.Y.Z/PythonAPI/carla/dist/carla-*-py3.*-linux-x86_64.egg",
          file=sys.stderr)
    raise

from inference import BCController  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to BC checkpoint (.pt)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town04")
    p.add_argument("--fps", type=int, default=20, help="Must match the training data's fps")
    p.add_argument("--vehicle", default="vehicle.tesla.model3")

    # Highway-only filtering (Town04 has a ring highway + inner city; we want the ring).
    p.add_argument("--highway-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Only spawn on multi-lane highway segments (default: True).")
    p.add_argument("--min-lanes", type=int, default=3,
                   help="Minimum parallel driving lanes to qualify as highway (default: 3).")
    p.add_argument("--min-speed-limit-kmh", type=float, default=60.0,
                   help="Optional secondary filter: require this speed limit on the road.")

    # Camera settings — MUST match what the model was trained on.
    p.add_argument("--cam-w", type=int, default=400)
    p.add_argument("--cam-h", type=int, default=300)
    p.add_argument("--cam-fov", type=float, default=90.0)
    p.add_argument("--cam-x", type=float, default=1.5)
    p.add_argument("--cam-y", type=float, default=0.0)
    p.add_argument("--cam-z", type=float, default=1.6)

    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    # Cold-start bootstrap: force throttle for N seconds to overcome engine
    # idle + spawn-settling. BC models trained on autopilot labels almost never
    # see a truly stationary-from-cold visual, so they can leave the vehicle
    # idling forever.
    p.add_argument("--bootstrap-seconds", type=float, default=2.0,
                   help="Apply a fixed throttle for N sim-seconds before handing to BC.")
    p.add_argument("--bootstrap-throttle", type=float, default=0.7)
    p.add_argument("--stuck-throttle-boost", action=argparse.BooleanOptionalAction, default=True,
                   help="If the ego stays < stuck-speed for stuck-ticks after bootstrap, "
                        "force throttle=0.8 until it moves.")
    p.add_argument("--stuck-speed-kmh", type=float, default=1.0)
    p.add_argument("--stuck-ticks", type=int, default=40)

    # Pre-flight: before handing over to BC, verify the chosen spawn actually
    # allows the vehicle to roll. If spawn-geometry pins it, try another.
    p.add_argument("--preflight-spawn-test", action=argparse.BooleanOptionalAction, default=True,
                   help="Test each spawn with constant throttle; keep the first that moves.")
    p.add_argument("--preflight-attempts", type=int, default=8)
    p.add_argument("--preflight-pass-kmh", type=float, default=3.0)
    return p.parse_args()


def carla_img_to_rgb(image: "carla.Image") -> np.ndarray:
    """CARLA returns BGRA uint8; model expects HxWx3 RGB uint8."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


def _count_parallel_driving_lanes(wp: "carla.Waypoint") -> int:
    """How many Driving lanes run in the same direction as this waypoint?"""
    if wp is None or wp.lane_type != carla.LaneType.Driving:
        return 0
    count = 1

    # Walk right.
    cursor = wp.get_right_lane()
    while cursor is not None and cursor.lane_type == carla.LaneType.Driving:
        count += 1
        cursor = cursor.get_right_lane()

    # Walk left — but only lanes going the SAME direction (lane_id sign match).
    cursor = wp.get_left_lane()
    while (cursor is not None
           and cursor.lane_type == carla.LaneType.Driving
           and (cursor.lane_id * wp.lane_id) > 0):
        count += 1
        cursor = cursor.get_left_lane()
    return count


def _teleport_reset(vehicle: "carla.Vehicle", transform: "carla.Transform", world: "carla.World"):
    """Move the car to a new transform and kill any residual velocity."""
    vehicle.set_simulate_physics(False)
    vehicle.set_transform(transform)
    vehicle.set_simulate_physics(True)
    try:
        vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    except Exception:
        pass  # older CARLA API names
    for _ in range(5):
        world.tick()


def _speed_kmh_of(vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5


def preflight_find_working_spawn(world, vehicle, spawn_points, fps,
                                 max_attempts, pass_kmh) -> "carla.Transform | None":
    """Try spawn points in random order; return the first one where full throttle
    actually produces motion. The vehicle is teleported between candidates."""
    attempts = min(max_attempts, len(spawn_points))
    candidates = random.sample(list(spawn_points), attempts)

    settle_ticks = fps  # 1 s
    drive_ticks = 2 * fps  # 2 s

    for i, sp in enumerate(candidates):
        _teleport_reset(vehicle, sp, world)
        for _ in range(settle_ticks):
            world.tick()
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0,
                                                       hand_brake=False))
        for _ in range(drive_ticks):
            world.tick()
            vehicle.apply_control(carla.VehicleControl(throttle=0.9, brake=0.0,
                                                       steer=0.0, hand_brake=False))
        speed = _speed_kmh_of(vehicle)
        loc = vehicle.get_location()
        print(f"preflight #{i}: {loc.x:+.1f},{loc.y:+.1f},z={loc.z:.2f}  "
              f"→ speed after {drive_ticks/fps:.1f}s @ thr=0.9: {speed:.1f} km/h")
        if speed >= pass_kmh:
            # Settle & brake to a stop before handing control back.
            for _ in range(fps // 2):
                world.tick()
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0,
                                                           hand_brake=False))
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0,
                                                       hand_brake=False))
            return sp
    return None


def filter_highway_spawns(world: "carla.World",
                          spawn_points: list,
                          min_lanes: int,
                          min_speed_kmh: float) -> list:
    """Keep spawn points that sit on multi-lane, high-speed-limit road segments.

    In Town04 this cleanly isolates the ring highway from the inner city:
    city streets almost never have 3+ parallel driving lanes, and city speed
    limits (30 km/h) are well below the highway (90 km/h).
    """
    carla_map = world.get_map()
    kept = []
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location,
                                    project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if _count_parallel_driving_lanes(wp) < min_lanes:
            continue

        # Secondary filter: speed-limit landmark near this spawn. Some CARLA
        # versions expose this; fail open when the API isn't available.
        if min_speed_kmh and min_speed_kmh > 0:
            try:
                landmarks = wp.get_landmarks_of_type(
                    50.0, "274"  # OpenDRIVE type code for speed-limit signs
                )
                if landmarks:
                    limits = [float(lm.value) for lm in landmarks if lm.value > 0]
                    if limits and max(limits) < min_speed_kmh:
                        continue
            except Exception:
                pass  # landmark API isn't always populated; don't exclude on that

        kept.append(sp)
    return kept


def main():
    args = parse_args()
    random.seed(args.seed)

    ctrl = BCController(args.checkpoint, device=args.device)
    print(f"Loaded BC controller from {args.checkpoint}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    # Load the requested town if it isn't already loaded — avoids a slow reload.
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != args.town:
        world = client.load_world(args.town)

    # Synchronous mode + fixed dt — critical for reproducible inference.
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    vehicle = None
    camera = None
    actors_to_destroy = []
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=4)

    # Ctrl-C handler to restore world settings.
    def cleanup(*_):
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
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    try:
        bp_lib = world.get_blueprint_library()

        # Spawn vehicle.
        vehicle_bp = bp_lib.filter(args.vehicle)[0]
        vehicle_bp.set_attribute("role_name", "hero")
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in this town.")

        if args.highway_only:
            highway_spawns = filter_highway_spawns(
                world, spawn_points,
                min_lanes=args.min_lanes,
                min_speed_kmh=args.min_speed_limit_kmh,
            )
            if not highway_spawns:
                raise RuntimeError(
                    f"No highway spawn points found in {args.town} "
                    f"(min_lanes={args.min_lanes}, min_speed={args.min_speed_limit_kmh} km/h). "
                    f"Loosen the filter with --min-lanes or pass --no-highway-only."
                )
            print(f"Highway filter: {len(highway_spawns)}/{len(spawn_points)} "
                  f"spawn points qualify in {args.town}.")
            spawn_points = highway_spawns

        spawn = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors_to_destroy.append(vehicle)
        print(f"Spawned {args.vehicle} at {spawn.location}")

        # Attach a collision sensor so we can tell if the vehicle is pinned
        # against a barrier (root cause of many "stuck" failures in Town04).
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors_to_destroy.append(collision_sensor)
        collision_hits = {"count": 0, "last_impulse": 0.0}

        def _on_collision(ev):
            i = ev.normal_impulse
            mag = (i.x ** 2 + i.y ** 2 + i.z ** 2) ** 0.5
            collision_hits["count"] += 1
            collision_hits["last_impulse"] = mag

        collision_sensor.listen(_on_collision)

        # Attach RGB camera that mirrors the training sensor exactly.
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.cam_w))
        cam_bp.set_attribute("image_size_y", str(args.cam_h))
        cam_bp.set_attribute("fov", str(args.cam_fov))
        cam_bp.set_attribute("sensor_tick", "0.0")
        cam_transform = carla.Transform(
            carla.Location(x=args.cam_x, y=args.cam_y, z=args.cam_z),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actors_to_destroy.append(camera)

        def _on_image(img):
            # Drop oldest if queue is full — we only care about the latest frame.
            if image_queue.full():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    pass
            image_queue.put(img)

        camera.listen(_on_image)

        # Warm up longer than before — elevated highway spawns in Town04 often
        # need more ticks to settle than city spawns.
        for _ in range(30):
            world.tick()

        # Pre-flight spawn test: bail out of spawn points that physically pin
        # the vehicle. Teleports between candidates until one produces motion.
        if args.preflight_spawn_test:
            good = preflight_find_working_spawn(
                world, vehicle, spawn_points,
                fps=args.fps,
                max_attempts=args.preflight_attempts,
                pass_kmh=args.preflight_pass_kmh,
            )
            if good is None:
                raise RuntimeError(
                    f"Pre-flight failed: no spawn in {args.preflight_attempts} attempts "
                    f"reached {args.preflight_pass_kmh} km/h under full throttle. "
                    f"Likely a physics/blueprint issue — try "
                    f"--vehicle vehicle.audi.a2 or --no-highway-only."
                )
            print(f"Pre-flight OK. Driving from {good.location}.")

        # Flush the image queue so we start from a fresh frame.
        while not image_queue.empty():
            try: image_queue.get_nowait()
            except queue.Empty: break

        def _speed_kmh() -> float:
            v = vehicle.get_velocity()
            return 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5

        bootstrap_ticks = int(args.bootstrap_seconds * args.fps)
        stuck_counter = 0
        mode = "bootstrap" if bootstrap_ticks > 0 else "bc"
        print(f"Starting in mode={mode} for {bootstrap_ticks} ticks, then BC.")

        step = 0
        t_end = time.time() + args.duration_s
        while time.time() < t_end:
            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("WARN: no camera frame received within 2s; aborting.")
                break

            rgb = carla_img_to_rgb(image)
            steer_bc, throttle_bc, brake_bc, throttle_raw, brake_raw = ctrl.act_with_raw(rgb)

            speed = _speed_kmh()

            if step < bootstrap_ticks:
                # Force motion from a cold spawn: model's steering is fine (road-
                # keeping signal is strong even at rest), but we override throttle.
                steer = steer_bc
                throttle = args.bootstrap_throttle
                brake = 0.0
                mode = "bootstrap"
            elif args.stuck_throttle_boost and speed < args.stuck_speed_kmh:
                stuck_counter += 1
                if stuck_counter >= args.stuck_ticks:
                    steer = steer_bc
                    throttle = 0.8
                    brake = 0.0
                    mode = "stuck-boost"
                else:
                    steer, throttle, brake = steer_bc, throttle_bc, brake_bc
                    mode = "bc"
            else:
                stuck_counter = 0
                steer, throttle, brake = steer_bc, throttle_bc, brake_bc
                mode = "bc"

            vehicle.apply_control(carla.VehicleControl(
                steer=float(steer),
                throttle=float(throttle),
                brake=float(brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            ))

            if step % args.fps == 0:  # once per simulated second
                loc = vehicle.get_location()
                ctrl_state = vehicle.get_control()
                print(f"step={step:5d} [{mode}] speed={speed:5.1f} km/h "
                      f"steer={steer:+.3f} thr={throttle:.3f} brk={brake:.3f} "
                      f"| raw thr={throttle_raw:.3f} brk={brake_raw:.3f} "
                      f"| gear={ctrl_state.gear} hb={ctrl_state.hand_brake} "
                      f"z={loc.z:.2f} coll={collision_hits['count']}"
                      f"{' J=%.0f' % collision_hits['last_impulse'] if collision_hits['count'] else ''}")
            step += 1

    finally:
        cleanup()


if __name__ == "__main__":
    main()
