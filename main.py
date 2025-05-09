import numpy as np
import torch
from flow.controllers import ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams
from flow.utils.registry import make_create_env
from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import MyEnv, ADDITIONAL_ENV_PARAMS
from utils import eval_policy_pareto_continuous
from agent import TD3
from memory import ReplayBuffer
import argparse
import os
import warnings
import pickle
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)                # Sets PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)         # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=3e5, type=int)     # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1, type=float)      # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)        # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)       # Discount factor
parser.add_argument("--tau", default=0.005, type=float)           # Target network update rate
parser.add_argument("--policy_noise", default=0.2)                # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)         # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")          # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                   # "" doesn't load, "default" uses file_name
parser.add_argument("--file_name", default="")
args = parser.parse_args()

nn_architecture = "smart"
vehicles = VehicleParams()
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=4,
             color='green')
sim_params = SumoParams(sim_step=0.1, render=False)
initial_config = InitialConfig()
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)

flow_params = dict(
    exp_tag='test_network',
    env_name=MyEnv,
    network=IntersectionNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

flow_params['env'].horizon = 1000
create_env, _ = make_create_env(flow_params)
env = create_env()
env.nn_architecture = nn_architecture
env.omega_space = "continuous"

file_name = f"aim_fair_{args.seed}_{args.file_name}"
print("---------------------------------------")
print(f"Seed: {args.seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

if args.save_model and not os.path.exists("./models"):
    os.makedirs("./models")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = 4
edge_dim = 2
action_dim = 1
max_action = 5.0
RED = (255, 0, 0)

aim = TD3(state_dim, edge_dim, action_dim, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
          noise_clip=args.noise_clip, policy_freq=args.policy_freq, max_action=max_action)
memory = ReplayBuffer()

if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    aim.load(f"./models/{policy_file}")
    _, _, _, _ = eval_policy_pareto_continuous(aim, env, nn_architecture=nn_architecture)
    env.terminate()
    raise KeyboardInterrupt

evaluations = []
fronts = []
crashes = []
num_crashes, hv, front, _ = eval_policy_pareto_continuous(aim, env, nn_architecture=nn_architecture)

evaluations.append(hv)
fronts.append(front)
crashes.append(num_crashes)
num_steps = env.env_params.horizon
num_evaluations = 6
best_hypervolume = 0
best_num_sol = 0

ep_steps = 0
ep_return = 0
ep_number = 0
state = env.reset()
while state.x is None:
    state, _, _, _ = env.step([])

ids = env.k.vehicle.get_ids()
elec_vehs = list(np.random.choice(ids, 2, replace=False))
env.k.vehicle.set_emission_class(elec_vehs)
for v in elec_vehs:
    env.k.vehicle.set_color(v, RED)

for t in range(int(args.max_timesteps)):

    ep_steps += 1

    om = env.omega
    if t < args.start_timesteps:
        actions = env.action_space.sample()
    else:
        actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                    torch.tensor([[env.omega, 1 - env.omega]],
                                                 dtype=torch.float, device=device).repeat(state.x.shape[0], 1))
        noise = np.random.normal(0.0, max_action * args.expl_noise, size=len(actions)).astype(np.float32)
        actions = (actions + noise).clip(-max_action, max_action)

    state_, reward, done, _ = env.step(rl_actions=actions)

    done_bool = float(done) if ep_steps < num_steps else 0.0

    if state_.x is None:
        memory.add(state, actions, state, reward, done_bool, om)
    else:
        memory.add(state, actions, state_, reward, done_bool, om)

    state = state_
    ep_return += reward

    if t >= args.start_timesteps:
        aim.train(memory, args.batch_size)

    if state.x is None:
        done = True

    if done:
        print(f"Total T: {t + 1} Episode Num: {ep_number + 1} Episode T: {ep_steps} Reward: {ep_return:.3f}")
        # Evaluate episode
        if (t + 1) >= args.eval_freq * num_evaluations and t >= 30000:
            num_crashes, hv, front, pareto_indexes = eval_policy_pareto_continuous(aim, env,
                                                                                   nn_architecture=nn_architecture)
            evaluations.append(hv)
            fronts.append(front)
            crashes.append(num_crashes)
            if args.save_model:
                with open(f"./results/{file_name}_hv.pkl", "wb") as f:
                    pickle.dump(evaluations, f)
                with open(f"./results/{file_name}_front.pkl", "wb") as f:
                    pickle.dump(fronts, f)
                with open(f"./results/{file_name}_crashes.pkl", "wb") as f:
                    pickle.dump(crashes, f)
            if num_crashes <= 50 and args.save_model:
                if len(front) > best_num_sol:
                    aim.save(f"./models/{file_name}")
                    best_hypervolume = evaluations[-1]
                    best_num_sol = len(front)
                    with open(f"./results/{file_name}_pareto_indexes.pkl", "wb") as f:
                        pickle.dump(pareto_indexes, f)
                if len(front) == best_num_sol and hv > best_hypervolume:
                    aim.save(f"./models/{file_name}")
                    best_hypervolume = evaluations[-1]
                    with open(f"./results/{file_name}_pareto_indexes.pkl", "wb") as f:
                        pickle.dump(pareto_indexes, f)
            num_evaluations += 1

        # Reset environment
        state = env.reset()
        while state.x is None:
            state, _, _, _ = env.step([])

        ids = env.k.vehicle.get_ids()
        elec_vehs = list(np.random.choice(ids, 2, replace=False))
        env.k.vehicle.set_emission_class(elec_vehs)
        for v in elec_vehs:
            env.k.vehicle.set_color(v, RED)

        ep_steps = 0
        ep_return = 0
        ep_number += 1

env.terminate()
