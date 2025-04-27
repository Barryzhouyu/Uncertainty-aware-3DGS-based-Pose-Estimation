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
                        "CaptureHeight": 1280,
                    }
                },
                {"sensor_type": "PoseSensor"},
                {"sensor_type": "LocationSensor"},
                {"sensor_type": "VelocitySensor"},
            ],
            "control_scheme": 0,
            "location": [30, -37.5, -292.5],
            "rotation": [0, 0, 180]
        }
    ],
}

def simple_rotate_and_forward():
    output_dir = "/home/roar/Desktop/LSTM_collected_data_simple/"
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

        env.tick(50)
        step = 0

        rotation_ticks = 0   
        forward_ticks = 300   
        yaw_speed = 0        

        def save_data(states, step):
            pose_matrix = np.array(states['auv0']["PoseSensor"])
            location = np.array(states['auv0']["LocationSensor"])
            xyz = [f"{x:.2f}" for x in location]
            filename = str(xyz) + ".png"
            image = states['auv0']['RGBCamera'][:, :, :3]
            cv2.imwrite(os.path.join(output_dir, filename), image)
            poses.append(pose_matrix)
            locations.append(location)
            timesteps.append(step)

        # 🔹 1. Rotate first
        print("Rotating...")
        for _ in range(rotation_ticks):
            states = env.tick()
            env.act('auv0', np.array([0, 0, 0, 0, 0, 0, 0, -yaw_speed]))
            step += 1  

        # 🔹 2. Move forward
        print("Moving forward...")
        for _ in range(forward_ticks):
            states = env.tick()
            agent = env.agents['auv0']
            pose_matrix = np.array(states['auv0']["PoseSensor"])
            location = np.array(states['auv0']["LocationSensor"])
            euler = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
            agent.set_physics_state(location=location, rotation=euler,
                                    velocity=np.array([-2, 0, 0], dtype='float32'),
                                    angular_velocity=np.zeros(3))
            if step % 10 == 0:
                save_data(states, step)
            step += 1

        # Save final data
        np.save(os.path.join(output_dir, "poses.npy"), np.array(poses))
        np.save(os.path.join(output_dir, "locations.npy"), np.array(locations))
        np.save(os.path.join(output_dir, "timesteps.npy"), np.array(timesteps))
        print("Data saved. Mission complete.")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    simple_rotate_and_forward()
