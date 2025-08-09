# logging the control commands:
# PX4 example using pymavlink
from pymavlink import mavutil
import csv

conn = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
with open('telemetry.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'throttle', 'yaw', 'pitch', 'roll'])
    
    while True:
        msg = conn.recv_match(type='RC_CHANNELS', blocking=True)
        writer.writerow([msg.time_boot_ms, 
                       msg.chan1_scaled, 
                       msg.chan2_scaled,
                       msg.chan3_scaled,
                       msg.chan4_scaled])