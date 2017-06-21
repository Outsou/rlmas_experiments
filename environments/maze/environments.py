from creamas.mp import MultiEnvironment
from creamas.vote import VoteOrganizer, vote_mean
from creamas.util import run

import aiomas
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np
import logging
import editdistance
import asyncio

class StatEnvironment(MultiEnvironment):
    def get_connection_counts(self):
        return self.get_dictionary('get_connection_counts')

    def get_comparison_counts(self):
        return self.get_dictionary('get_comparison_count')

    def get_artifacts_created(self):
        return self.get_dictionary('get_artifacts_created')

    def get_dictionary(self, func_name):
        agents = self.get_agents(addr=False)

        dict = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            func = getattr(agent, func_name)
            dict[name] = aiomas.run(until=func())

        return dict

class MazeMultiEnvironment(StatEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.age = 0
        self.voting_method = vote_mean
        self.valid_cand = []
        self.suggested_cand = []
        logger = logging.getLogger('mazes.vo')
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.DEBUG)
        self.vote_organizer = VoteOrganizer(self, logger=logger)



    def vote_and_save_info(self, age):
        self.age = age
        self.vote_organizer.gather_candidates()
        self.suggested_cand.append(len(self.vote_organizer.candidates))
        self.vote_organizer.validate_candidates()
        self.valid_cand.append(len(self.vote_organizer.candidates))
        self.vote_organizer.gather_votes()
        artifacts = self.vote_organizer.compute_results(voting_method=self.voting_method)
        threshold = 0.0

        for a,v in artifacts:
            accepted = True if v >= threshold else False
            a.accepted = accepted
            self.add_artifact(a)
            tasks = []
            for addr in self._manager_addrs:
                tasks.append(asyncio.ensure_future(self._add_domain_artifact(addr, a)))
            aiomas.run(until=asyncio.gather(*tasks))

        self.vote_organizer.clear_candidates(clear_env=True)
        self.valid_candidates = []

    def save_domain_artifacts(self, folder):
        self._calc_distances()

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

    def _calc_distances(self):
        accepted_x = []
        accepted_y = []
        rejected_x = []
        rejected_y = []
        sort_arts = sorted(self.artifacts, key=lambda x: x.env_time)

        for i,a1 in enumerate(sort_arts[1:]):
            solution1 = a1.obj['solution']
            i = i+1
            mdist = np.inf
            for a2 in sort_arts[:i]:
                solution2 = a2.obj['solution']
                dist = editdistance.eval(solution1, solution2)
                if dist < mdist:
                    mdist = dist
            if a1.accepted:
                accepted_x.append(a1.env_time)
                accepted_y.append(mdist)
            else:
                rejected_x.append(a1.env_time)
                rejected_y.append(mdist)
        mean_dist = np.mean(accepted_y)
        self._log(logging.INFO, "Mean of (accepted) distances: {}".format(mean_dist))
        return mean_dist, (accepted_x, accepted_y), (rejected_x, rejected_y)

    async def _add_domain_artifact(self, manager_addr, artifact):
        pass

class GatekeeperMazeMultiEnvironment(StatEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gatekeepers = []

    def publish_artifacts(self):
        for gatekeeper in self.gatekeepers:
            run(gatekeeper.publish())
