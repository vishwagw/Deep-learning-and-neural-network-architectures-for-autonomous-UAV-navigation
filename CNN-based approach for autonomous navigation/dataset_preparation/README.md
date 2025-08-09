# Data Collection Setup

## Hardware Requirements:

Drone with a high-quality camera (minimum 1080p @ 30FPS)

GPS/IMU module (for precise telemetry logging)

Onboard computer (e.g., Raspberry Pi or Jetson) or ground station recording

## Software Requirements:

Video recording software (e.g., ffmpeg, ROS image_view)

Telemetry logger (e.g., MAVLink for PX4/ArduPilot drones)

Synchronization tool (e.g., time-sync scripts)

# Training data:

## flight data recording:

### On drone computer (example for Raspberry Pi)
ffmpeg -i /dev/video0 -vf "fps=30" -c:v h264 flight1.mp4

## then use control_cmd_log.py for recording telemetry data.

# Data set structure:

/drone_dataset
│── /raw
│   ├── flight1.mp4
│   ├── flight1_telemetry.csv
│   ├── flight2.mp4
│   └── ...
│── /processed
│   ├── train
│   │   ├── images (folder of individual frames)
│   │   └── labels.csv
│   └── test
│       ├── images
│       └── labels.csv
└── metadata.json (contains FPS, resolution, etc.)

# 