import os
import argparse
import json
import gym
from agent import SACAgent


def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    config["locexp"] = args.locexp
    path = args.locexp
    # experiment_name = args.experiment_name
    vid_path = os.path.join(path, "videos-{}".format(args.seed))
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    config["vid_path"] = vid_path
    config["res_path"] = res_path
    config["seed"] = args.seed
    env = gym.make(config["env_name"])
    print(str(config))
    action_size = env.action_space.n
    print(action_size)
    state_size = 200
    agent = SACAgent(action_size=action_size, state_size=state_size, config=config)
    agent.train_agent()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="LunarLanderContinuous-v2", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--locexp', default="test", type=str)
    parser.add_argument('--param', default="param.json", type=str)
    arg = parser.parse_args()
    main(arg)
