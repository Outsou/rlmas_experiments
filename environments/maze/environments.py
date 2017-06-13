from creamas.mp import MultiEnvironment

import aiomas
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np


class MazeMultiEnvironment(MultiEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = 0
        self.voting_method = 'mean'
        self.valid_cand = []
        self.suggested_cand = []

    async def get_candidates(self, addr):
        candidates =  await self._manager.get_candidates(addr)
        return candidates

    async def gather_candidates(self):
        cands = []
        for addr in self._manager_addrs:
            cand = await self.get_candidates(addr)
            cands.extend(cand)
        return cands

    def get_connection_counts(self):
        return self.get_dictionary('get_connection_counts')

    def get_dictionary(self, func_name):
        agents = self.get_agents(addr=False)

        dict = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            func = getattr(agent, func_name)
            dict[name] = aiomas.run(until=func())

        return dict


    def vote_and_save_info(self, age):
        self.age = age
        self._candidates = aiomas.run(until=self.gather_candidates())
        self.suggested_cand.append(len(self.candidates))
        self.validate_candidates()
        self.valid_cand.append(len(self.candidates))
        artifacts = self.perform_voting(method=self.voting_method)
        threshold = 0.0

        for a, v in artifacts:
            accepted = True if v >= threshold else False
            a.accepted = accepted
            self.add_artifact(a)

        self.clear_candidates()
        self.valid_candidates = []

    def save_domain_artifacts(self, folder):
        for artifact in self.artifacts:
            value = 0.7
            maze = artifact.obj['maze']

            masked_array = np.ma.masked_where(maze == value, maze)

            cmap = matplotlib.cm.gray
            cmap.set_bad(color='blue')

            fig, ax = plt.subplots(1)
            ax.imshow(masked_array, cmap=cmap, interpolation=None)
            start = artifact.obj['start']
            start_circle = patches.Circle((start[1], start[0]), 2, linewidth=2, edgecolor='y', facecolor='none')
            ax.add_patch(start_circle)
            goal = artifact.obj['goal']
            goal_circle = patches.Circle((goal[1], goal[0]), 2, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(goal_circle)

            plt.savefig('{}/maze_{}.png'.format(folder, artifact.env_time))
            plt.close()
