from creamas.examples.spiro.spiro_agent_mp import SpiroMultiEnvironment
import aiomas
import asyncio
import logging


class SprEnvironment(SpiroMultiEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_agent_acquaintances(self):
        agents = self.get_agents(address=False)
        addresses = self.get_agents(address=True)

        for agent in agents:
            aiomas.run(until=agent.set_acquaintances(addresses))

    def get_total_reward(self):
        agents = self.get_agents(address=False)

        total_reward = 0

        for agent in agents:
            total_reward += aiomas.run(until=agent.get_total_reward())

        return total_reward

    def log_situation(self, step):
        agents = self.get_agents(address=False)

        for agent in agents:
            aiomas.run(until=agent.log_situation())

    def get_acquaintance_counts(self):
        agents = self.get_agents(address=False)

        acquaintances = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            acquaintances[name] = aiomas.run(until=agent.get_acquaintances())

        return acquaintances

    def get_acquaintance_values(self):
        agents = self.get_agents(address=False)

        acquaintance_values = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            acquaintance_values[name] = aiomas.run(until=agent.get_acquaintance_values())

        return acquaintance_values

    def get_comparison_count(self):
        agents = self.get_agents(address=False)

        total_comparisons = 0

        for agent in agents:
            total_comparisons += aiomas.run(until=agent.get_comparison_count())

        return total_comparisons

    def destroy(self, folder=None):
        '''Destroy the environment and the subprocesses.
        '''
        #ameans = [(0, 0, 0) for _ in range(3)]
        #ret = [self.save_info(folder, ameans)]
        rets = aiomas.run(until=self._destroy_slaves(folder))
        #rets = ret + rets
        # Close and join the process pool nicely.
        self._pool.close()
        self._pool.terminate()
        self._pool.join()
        self._env.shutdown()
        return rets

    def validate_candidates(self):
        '''Validate current candidates in the environment by pruning candidates
        that are not validated at least by one agent, i.e. they are vetoed.

        In larger societies this method might be costly, as it calls each
        agents' ``validate_candidates``-method.
        '''

        valid_candidates = set(self.candidates)
        tasks = []
        for a in self._manager_addrs:
            tasks.append(self._validate_candidates(a))
        ret = aiomas.run(until=asyncio.gather(*tasks))

        for r in ret:
            result = aiomas.run(until=r)
            vc = set(result)
            valid_candidates = valid_candidates.intersection(vc)

        self._candidates = list(valid_candidates)
        self._log(logging.INFO,
                  "{} valid candidates after get_agents used veto."
                  .format(len(self.candidates)))