# Husky UR5 robot

This package includes two parts: RL part and ros part. The RL part gets current state from simulation/real, and predicts the next action.

## Simulation



## Real 

For deploying the trained policy to real robot, we use ros as a middleware to communicate with the robot. You can try two different ways to control: MoveIt! or actionlib. By default, we use MoveIt! to control the ur with a rrt motion planning. However, it is hard to get a continuous trajectory if using a position control style. So we use the ros_control as the ur5 ros driver and a position controller to communicate with it. 

ros environment: only left the basic part for it, such as step, action, observation.


## Problems

1. When the robot close to the object and cannot see it again, how to deal with it??? Use the last object position and stop the husky.

2. Base and arm's control frequence is 10Hz, but the gripper is different. How to deal with it? When try to close the gripper, set the base and arm action is 0.

3. We should let the husky move to somewhere that the arm can grasp the object, then set the base action to 0. Add a distance of object and the center of robot to judge it.

4. 