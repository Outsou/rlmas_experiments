from creamas.mp import MultiEnvironment
from creamas.vote import VoteOrganizer, vote_mean
from creamas.util import run
import mazes.maze_solver as ms

import aiomas
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np
import logging
import editdistance
import asyncio
import os
import shutil


class StatEnvironment(MultiEnvironment):
    def get_connection_counts(self):
        return self.get_dictionary('get_connection_counts')

    def get_comparison_counts(self):
        return self.get_dictionary('get_comparison_count')

    def get_artifacts_created(self):
        return self.get_dictionary('get_artifacts_created')

    def get_passed_self_criticism_counts(self):
        return self.get_dictionary('get_passed_self_criticism_count')

    def get_dictionary(self, func_name):
        agents = self.get_agents(addr=False)

        dict = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            func = getattr(agent, func_name)
            dict[name] = aiomas.run(until=func())

        return dict

    @staticmethod
    def save_artifact(artifact, folder, id, eval=None):
        value = 0.7
        maze = artifact.obj['maze']
        maze = ms.draw_path(maze, artifact.obj['path'], value)

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

        if eval is not None:
            plt.title('Evaluation: {}'.format(eval))
        plt.savefig('{}/maze_{}.png'.format(folder, id))
        plt.close()


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
            self.save_artifact(artifact, folder, artifact.env_time)

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

    def publish_artifacts(self, age):
        for gatekeeper in self.gatekeepers:
            run(gatekeeper.publish())

    def get_choose_func_counts(self):
        return self.get_creator_dictionary('get_choose_func_counts')

    def get_published_counts(self):
        return self.get_creator_dictionary('get_published_count')

    def get_func_values(self):
        return self.get_creator_dictionary('get_func_values')

    def get_published_creators(self):
        return self.get_gatekeeper_dictionary('get_published_creators')

    def get_creator_dictionary(self, func_name):
        agents = self.get_agents(addr=False)

        dict = {}

        for agent in agents:
            if agent not in self.gatekeepers:
                name = run(agent.get_name())
                func = getattr(agent, func_name)
                dict[name] = run(func())

        return dict

    def get_gatekeeper_dictionary(self, func_name):
        dict = {}

        for agent in self.gatekeepers:
            name = run(agent.get_name())
            func = getattr(agent, func_name)
            dict[name] = run(func())

        return dict

    def save_creator_artifacts(self, folder):
        agents = self.get_agents(addr=False)

        for agent in agents:
            if agent not in self.gatekeepers:
                id = 0
                name = run(agent.get_name())
                parsed_name = name.replace('://', '_')
                parsed_name = parsed_name.replace(':', '_')
                parsed_name = parsed_name.replace('/', '_')
                agent_folder = folder + '/' + parsed_name

                if os.path.exists(agent_folder):
                    shutil.rmtree(agent_folder)
                os.makedirs(agent_folder)

                artifacts = run(agent.get_memory_artifacts())

                for artifact in artifacts:
                    id += 1
                    self.save_artifact(artifact, agent_folder, id, artifact.evals[name])