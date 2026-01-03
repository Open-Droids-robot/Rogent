#!/bin/bash

# Define ROS commands
CMD_T1="source /opt/ros/foxy/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 launch ros2_total_demo total_demo.launch.py"
CMD_T2="source /opt/ros/foxy/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
# Note: Do not use VS Code / Cursor terminal for this command, it will cause errors. Use a regular terminal.

# Define agent commands; change to agent.py for the old version
# Note: Using direct path to python in venv is often more reliable in scripts than sourcing
CMD_AGENT="source .venv/bin/activate && python src/agent_v2.py"

# Cleanup function to kill background processes when script is stopped
cleanup() {
    echo ""
    echo "Stopping all processes..."
    if [ -n "$PID1" ]; then kill $PID1 2>/dev/null; fi
    if [ -n "$PID2" ]; then kill $PID2 2>/dev/null; fi
    exit
}

# Trap Ctrl+C (SIGINT) and termination signal (SIGTERM)
trap cleanup SIGINT SIGTERM

echo "Starting Total Demo (background)..."
bash -c "$CMD_T1" &
PID1=$!
echo "Started Total Demo with PID: $PID1"

echo "Starting Rosbridge (background)..."
bash -c "$CMD_T2" &
PID2=$!
echo "Started Rosbridge with PID: $PID2"

# Wait a moment for ROS to initialize
sleep 5

# Start agent
echo "Starting Agent..."
bash -c "$CMD_AGENT"

# When agent exits, cleanup ROS nodes
cleanup
