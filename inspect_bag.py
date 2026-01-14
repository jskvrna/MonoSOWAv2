from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import stores, get_typestore

bag_path = Path('/path/to/DSEC/lidar/interlaken_00/lidar_imu.bag')

with AnyReader([bag_path]) as reader:
    print("Connections:")
    for c in reader.connections:
        print(f"  {c.topic} : {c.msgtype}")
    
    print("\nFirst 5 messages:")
    count = 0
    for connection, timestamp, rawdata in reader.messages():
        print(f"  Topic: {connection.topic}, Timestamp: {timestamp}")
        count += 1
        if count >= 5:
            break
