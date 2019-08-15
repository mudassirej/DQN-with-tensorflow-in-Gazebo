# DQN-with-tensorflow-in-Gazebo
Autonomous visual navigation using the depth images
git clone https://github.com/mudassirej/DQN-with-tensorflow-in-Gazebo.git
cd DQN-with-tensorflow-in-gazebo
roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/.test.world
python DQN_training.py
