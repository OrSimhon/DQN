import torch


def log_summary(ep_ret, ep_num):
    """
    Print to stdout what we have logged so far in the most recent episode.
    :param ep_ret: None
    :param ep_num: None
    :return: None
    """
    # Round decimal places for more aesthetic logging messages
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print("------------------------- Episode #{} -------------------------".format(ep_num), flush=True)
    print("Episodic Return: {}".format(ep_ret), flush=True)
    print("---------------------------------------------------------------")
    print(flush=True)


def rollout(policy, env, render):
    """
    Returns a generator to roll out each episode given a trained policy and environment to test on
    :param policy: Policy to test
    :param env: The environment to evaluate the policy on
    :param render: Specifies whether to render or not
    :return: A generator object rollout,
    """
    # Rollout until user kills process
    while True:
        obs = env.reset()
        done = False

        # number of timesteps so far
        t = 0

        # Logging data
        ep_ret = 0  # Episodic return

        while not done:
            t += 1

            # Render environment if specified, off by default
            if render:
                env.render()

            # Query deterministic action from policy and run it
            action = torch.argmax(policy(obs).detach()).item()
            obs, rew, done, _ = env.step(action)

            # Sum all episodic rewards as we go along
            ep_ret += rew

        # returns episodic length and return in this iteration
        yield ep_ret


def eval_policy(policy, env, render=False):
    """
    The main function to evaluate our policy with. It will iterate a generator object "rollout", which will simulate each
    episode and return the most recent episode's length and return. We can then log it right after. And yes, eval_policy will
    run forever until you kill the process.
    :param policy: The trained policy to test, basically another name for our actor model
    :param env: The environment to test the policy on
    :param render: Whether we should render our episodes. False by default
    :return: None
    """
    # Rollout with the policy and environment, and log each episode's data
    for ep_num, ep_ret in enumerate(rollout(policy, env, render)):
        log_summary(ep_ret=ep_ret, ep_num=ep_num)
