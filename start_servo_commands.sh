#!/bin/bash

# Define commands for each tab
# Tab 1: Total Demo
CMD_T1="source /opt/ros/foxy/setup.bash && source install/setup.bash && ros2 launch ros2_total_demo total_demo.launch.py"

# Tab 2: Rosbridge
CMD_T2="source /opt/ros/foxy/setup.bash && source install/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml"

# Check for gnome-terminal (common on Ubuntu ROS setups)
if command -v gnome-terminal &> /dev/null; then
    echo "Launching commands in gnome-terminal tabs..."
    gnome-terminal --window \
        --tab --title="Total Demo" -- bash -c "$CMD_T1; exec bash" \
        --tab --title="Rosbridge" -- bash -c "$CMD_T2; exec bash"
else
    echo "gnome-terminal not found."
    echo "Attempting to run commands in background (output will be interleaved)..."
    
    echo "Starting Tab 1 (Total Demo)..."
    bash -c "$CMD_T1" &
    
    echo "Starting Tab 2 (Rosbridge)..."
    bash -c "$CMD_T2" &
    
    wait
fi

