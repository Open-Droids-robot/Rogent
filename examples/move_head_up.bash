#!/bin/bash
# This script launches the total_demo ROS2 node in the background, then sends a command
# to move the robot's head up by publishing an appropriate message to the /servo_control/move topic.
# It logs background process output, sets up safe cleanup on termination, and keeps running until stopped.

# Define commands
CMD_T1="source /opt/ros/foxy/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 launch ros2_total_demo total_demo.launch.py"

# Cleanup function to kill background processes when script is stopped
cleanup() {
    echo ""
    echo "Stopping ROS nodes..."
    if [ -n "$PID1" ]; then kill $PID1 2>/dev/null; fi
    exit
}

# Trap Ctrl+C (SIGINT) and termination signal (SIGTERM)
trap cleanup SIGINT SIGTERM

echo "Starting Total Demo (background)... Logging to total_demo.log"
bash -c "$CMD_T1" > total_demo.log 2>&1 &
PID1=$!
echo "Started Total Demo with PID: $PID1"

# Small delay to let the first node start up
sleep 2

echo "Moving head up..."
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic pub --once /servo_control/move servo_interfaces/msg/ServoMove "{servo_id: 1, angle: 900}"

echo "---------------------------------------------------"
echo "Processes running in background."
echo "Press Ctrl+C to stop all processes."
echo "---------------------------------------------------"

# Wait for processes to finish (keeps script running so trap works)
wait
