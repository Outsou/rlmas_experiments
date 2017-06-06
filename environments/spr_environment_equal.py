from environments.spr_environment import SprEnvironment

import aiomas
import asyncio

class SprEnvironmentEqual(SprEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote_and_save_info(self, age):
        # self.age = age
        # self._candidates = aiomas.run(until=self.gather_candidates())
        # self.suggested_cand.append(len(self.candidates))
        # self.validate_candidates()
        # self.valid_cand.append(len(self.candidates))
        # artifacts = self.perform_voting(method=self.voting_method)
        # threshold = 0.0
        #
        # for a, v in artifacts:
        #     accepted = True if v >= threshold else False
        #     a.accepted = accepted
        #     self.add_artifact(a)
        #     tasks = []
        #     for addr in self._manager_addrs:
        #         tasks.append(asyncio.ensure_future(self._add_domain_artifact(addr, a)))
        #     aiomas.run(until=asyncio.gather(*tasks))
        #
        # self.clear_candidates()
        # self.valid_candidates = []

        super().vote_and_save_info(age)

        agents = self.get_agents(addr=False)
        self._consistent = False
        for agent in agents:
            aiomas.run(until=agent.process_rewards())
