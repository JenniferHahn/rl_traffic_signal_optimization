import os
import sys

import gym
import numpy as np

import platform
# on the server (platform = Linux) we use libsumo and also don't need the tools in the path
# if platform.system() != "Linux":
#     if 'SUMO_HOME' in os.environ:
#         tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#         sys.path.append(tools)  # we need to import python modules from the $SUMO_HOME/tools directory
#     else:
#         sys.exit("please declare environment variable 'SUMO_HOME'")
#     import traci
# else:
import libsumo as traci

from sumolib import checkBinary

from stable_baselines3.ppo.ppo import PPO

#from sumo_rl import SumoEnvironment
from environment.env import SumoEnvironment


env = SumoEnvironment(
    net_file="urban_mobility_simulation/models/20230502_SUMO_MA/osm.net.xml, \
            urban_mobility_simulation/models/20230502_SUMO_MA/pt/stops.add.xml, \
            urban_mobility_simulation/models/20230502_SUMO_MA/osm.poly.xml",
    single_agent=True,
    route_file="urban_mobility_simulation/models/20230502_SUMO_MA/routes.xml, \
               urban_mobility_simulation/models/20230502_SUMO_MA/osm.bicycle.trips.xml,\
            urban_mobility_simulation/models/20230502_SUMO_MA/osm.motorcycle.trips.xml,\
                urban_mobility_simulation/models/20230502_SUMO_MA/osm.truck.trips.xml, \
               urban_mobility_simulation/models/20230502_SUMO_MA/pt/ptflows.rou.xml, \
                urban_mobility_simulation/models/20230502_SUMO_MA/osm.passenger.trips.xml", 
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_orig_43200",
    use_gui=False,
    num_seconds=20000,
    yellow_time=4,
    min_green=5,
    max_green=60,
    time_to_teleport=300,
    fixed_ts=False,
#    additional_sumo_cmd="--scale 0.5",
#    begin_time=10000,
)

model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    verbose=1,
#    gamma=0.95,
#    gae_lambda=0.9,
#    ent_coef=0.1,
#    vf_coef=0.25,
#    batch_size=64
)
# Paths:
save_path = 'PPO_20000_orig'
log_path = 'Logs'

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

model.learn(total_timesteps=20000)

model.save(save_path)


