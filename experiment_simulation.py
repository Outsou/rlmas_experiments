from creamas.core.simulation import Simulation

import logging
import time


class ExperimentSimulation(Simulation):

    def __init__(self, environment, num_of_sim,  *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.num_of_sim = num_of_sim

    def _init_step(self):
        '''Initialize next step of simulation to be run.'''
        self._age += 1
        self.env.age = self._age
        self._log(logging.INFO, "")
        self._log(logging.INFO, "\t***** Sim  {:0>4} *****".format(self.num_of_sim))
        self._log(logging.INFO, "\t***** Step {:0>4} *****". format(self.age))
        self._log(logging.INFO, "")
        self._agents_to_act = self._get_order_agents()
        self._step_processing_time = 0.0
        self._step_start_time = time.time()