from agents.critic_equal_agent import CriticEqualAgent
from rl.q_learner import QLearner

import aiomas


class CriticQAgent(CriticEqualAgent):

    def __init__(self, environment, discount_factor, learning_factor, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)

        # self.memory_learner = QLearner(states=len(self.memory_states),
        #                                actions=len(self.memory_states),
        #                                discount_factor=discount_factor,
        #                                learning_factor=learning_factor,
        #                                initial_values=initial_values)

        self.discount_factor = discount_factor
        self.learning_factor =learning_factor

    @aiomas.expose
    def set_acquaintances(self, addresses):
        super().set_acquaintances(addresses)
        self.bandit_learner.set_values(-5)

    @aiomas.expose
    def process_rewards(self):
        '''Called after voting, so agent can process all the reward gained at once'''
        # Record in which state this time step was spent
        self.memory_state_times[self.current_memory_state] += 1

        # Reward is the sum of creation and criticism rewards of this time step
        self.memory_learner.give_reward_q_style(self.current_memory_state,
                                                self.creation_reward + self.criticism_reward,
                                                self.discount_factor,
                                                self.learning_factor)

        # Set rewards to zero for next time step
        self.creation_reward = 0
        self.criticism_reward = 0

        # Choose next memory state
        self.current_memory_state = self.memory_learner.choose_bandit()
        self.stmem.length = self.memory_states[self.current_memory_state]

    @aiomas.expose
    async def act(self):
        # Invent artifact using constant amount of comparisons
        invent_n = self.invent_n / self.memory_states[self.current_memory_state]
        artifact = self.invent(int(invent_n))

        self.added_last = False

        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self.spiro_args = args
        self.arg_history.append(self.spiro_args)
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            self.overcame_own_threshold_count += 1
            artifact.self_criticism = 'pass'
            # Train SOM with the invented artifact
            self.learn(artifact, self.teaching_iterations)

            # Check with another agent if the artifact is novel enough
            if self.ask_passing:
                # Ask someone for veto
                bandit = self.bandit_learner.choose_bandit(rand=self.rand)
                acquaintance = self.acquaintances[bandit]
                acquaintance[1] += 1

                connection = await self.env.connect(acquaintance[0])
                passed, artifact = await connection.ask_if_passes(artifact)

                if passed:
                    self.bandit_learner.give_reward_q_style(bandit, -1, self.discount_factor, self.learning_factor)
                    self.env.add_candidate(artifact)
                    self.added_last = True
                else:
                    self.bandit_learner.give_reward_q_style(bandit, 1, self.discount_factor, self.learning_factor)
            else:
                self.env.add_candidate(artifact)
                self.added_last = True
        elif self.jump == 'random':
            largs = self.spiro_args
            self.spiro_args = np.random.uniform(-199, 199,
                                                self.spiro_args.shape)
            self._log(logging.DEBUG, "Jumped from {} to {}"
                      .format(largs, self.spiro_args))

    @aiomas.expose
    def reset(self):
        # reset acquaintance counts
        for acquaintance in self.acquaintances:
            acquaintance[1] = 0

        self.comparison_count = 0
        self.overcame_own_threshold_count = 0
        self.rejection_count = 0
        self.opinion_asked_count = 0
        self.memory_state_times = [0] * len(self.memory_states)

