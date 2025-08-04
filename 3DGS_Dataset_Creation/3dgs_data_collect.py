import holoocean
import cv2
import os
import uuid
import numpy as np
from scipy.spatial.transform import Rotation as R

base_cfg = {
    "name": "data collection",
    "world": "OpenWater",
    "main_agent": "auv0",
    "frames_per_sec": False,
    "ticks_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv1",
            "agent_type": "SphereAgent",
            "sensors": [{"sensor_type": "ViewportCapture"}],
            "control_scheme": 0,
            "location": [0, 3, 100],
        },
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "socket": "CameraRightSocket",
                    "configuration": {
                        "CaptureWidth": 1280,
                        "CaptureHeight": 720,
                    }
                },
                {"sensor_type": "PoseSensor"},
                {"sensor_type": "LocationSensor"},
                {"sensor_type": "VelocitySensor"},
            ],
            "control_scheme": 0,
            "location": [40, -40, -290],
            "rotation": [0, 0, 120]
        }
    ],
}

def sporadic_observation():
    output_dir = "/home/roar/Desktop/LSTM_collected_data/"
    os.makedirs(output_dir, exist_ok=True)

    poses = []
    locations = []
    timesteps = []

    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=base_cfg,
        binary_path=binary_path,
        show_viewport=True,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:

        env.tick(100)
        step = 0

        rotation_ticks = 10  # ~1 second
        forward_ticks = 20  # ~2 seconds per segment
        yaw_speed = 0
        yaw_speed_2 = 0

        def save_data(states, step):
            pose_matrix = np.array(states['auv0']["PoseSensor"])
            location = np.array(states['auv0']["LocationSensor"])
            xyz = [f"{x:.2f}" for x in location]
            filename = str(xyz) + ".png"
            image = states['auv0']['RGBCamera'][:, :, :3]
            cv2.imwrite(os.path.join(output_dir, filename), image)
            cv2.imshow("Camera Output", image)
            cv2.waitKey(1)
            poses.append(pose_matrix)
            locations.append(location)
            timesteps.append(step)

        print("Running phase: repeating phase loop for 1 minute")
        total_ticks = 60 * 15  # 1 minute

        while step < total_ticks:

            for _ in range(forward_ticks):
                states = env.tick()
                agent = env.agents['auv0']
                pose_matrix = np.array(states['auv0']["PoseSensor"])
                location = np.array(states['auv0']["LocationSensor"])
                euler = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
                agent.set_physics_state(location=location, rotation=euler,
                                        velocity=np.array([-2, 0, 0], dtype='float32'),
                                        angular_velocity=np.zeros(3))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1

                # Turn right
            for _ in range(rotation_ticks):
                states = env.tick()
                env.act('auv0', np.array([0, 0, 0, 0, 0, 0, 0, -yaw_speed]))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1

            # Move forward + up
            for _ in range(forward_ticks):
                states = env.tick()
                agent = env.agents['auv0']
                pose_matrix = np.array(states['auv0']["PoseSensor"])
                location = np.array(states['auv0']["LocationSensor"])
                euler = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
                agent.set_physics_state(location=location, rotation=euler,
                                        velocity=np.array([-1.5, 0, 0], dtype='float32'),
                                        angular_velocity=np.zeros(3))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1
                
            # Turn left to return to original yaw
            for _ in range(rotation_ticks):
                states = env.tick()
                env.act('auv0', np.array([0, 0, 0, 0, 0, 0, 0, -yaw_speed_2]))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1

            # Move forward + down
            for _ in range(forward_ticks):
                states = env.tick()
                agent = env.agents['auv0']
                pose_matrix = np.array(states['auv0']["PoseSensor"])
                location = np.array(states['auv0']["LocationSensor"])
                euler = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
                agent.set_physics_state(location=location, rotation=euler,
                                        velocity=np.array([-1.5, 0, 0], dtype='float32'),
                                        angular_velocity=np.zeros(3))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1


            # Additional left turn and forward motion to complete loop
            for _ in range(rotation_ticks):
                states = env.tick()
                env.act('auv0', np.array([0, 0, 0, 0, 0, 0, 0, -yaw_speed_2]))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1

            for _ in range(forward_ticks):
                states = env.tick()
                agent = env.agents['auv0']
                pose_matrix = np.array(states['auv0']["PoseSensor"])
                location = np.array(states['auv0']["LocationSensor"])
                euler = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
                agent.set_physics_state(location=location, rotation=euler,
                                        velocity=np.array([-1.5, 0, 0], dtype='float32'),
                                        angular_velocity=np.zeros(3))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1

            # Turn right to restore original heading
            for _ in range(rotation_ticks):
                states = env.tick()
                env.act('auv0', np.array([0, 0, 0, 0, 0, 0, 0, -yaw_speed]))
                if step % 30 == 0:
                    save_data(states, step)
                step += 1

        # End of loop
        np.save(os.path.join(output_dir, "poses.npy"), np.array(poses))
        np.save(os.path.join(output_dir, "locations.npy"), np.array(locations))
        np.save(os.path.join(output_dir, "timesteps.npy"), np.array(timesteps))
        cv2.destroyAllWindows()
        print("Symmetric turn + forward-up-down phase complete and data saved.")

if __name__ == "__main__":
    sporadic_observation()



