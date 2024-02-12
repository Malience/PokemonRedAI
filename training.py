from rollout import generate_rollouts_vec, compose_rollouts

class Trainer:
    def train(self, obs, actions, action_log_probs, values, advantages, returns, batch_size, verbose=False):
        pass
    
def train(policies, env, num_envs, target, trainer: Trainer, num_steps=400, num_iterations=100, verbose=False):
    for iteration in range(1, num_iterations + 1):
        #rollouts = generate_rollouts(policies, main_env, 'explore_agent', count=num_envs, max_steps=num_steps, verbose=True)
        rollouts = generate_rollouts_vec(policies, env, target, count=num_envs, max_steps=num_steps, verbose=True)

        successes = sum([1 if rollout.success else 0 for rollout in rollouts[target]])
        total = len(rollouts[target])
        print(f"Success Rate: {successes/total*100}%")

        if successes == 0:
            print("Repeating without training!")
            #continue
        
        train_rollouts = []
        for rollout in rollouts[target]:
            #if rollout.success: 
                train_rollouts.append(rollout)

        compiled = compose_rollouts(rollouts[target])

        trainer.train(*compiled, batch_size=num_envs * num_steps, verbose=True)