# Use rllib to train the husky ur5

In the file `rllib_train.py`, we add a custom metric `success rate`. You should change the parameter `"timesteps_total": 4000000` and `"num_workers": 25`. Run the command:

```bash
(gym-env)$ cd ~/ros_workspace/openai_ros_ws/gym
python gym/envs/robotics/algos/rllib_train.py --run PPO/DDPG/A2C
```

Evaluate the policy:
```bash
rllib rollout path/to/checkpoint --run PPO
```