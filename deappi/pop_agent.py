from creamas.core.agent import CreativeAgent

import aiomas
import numpy as np

from deap import base
from deap import creator
from deap import gp
from deap import tools


class PopAgent(CreativeAgent):
    def __init__(self, env, pset, mate_func, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.pset = pset
        self.pop = self.create_pop(10)
        self.toolbox = self.create_toolbox(mate_func)

    def create_pop(self, size):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                       pset=self.pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        return toolbox.population(n=size)

    def evaluate(self, individual):
        return np.random.random(),

    def print_name(self):
        print(self.name)

    def create_toolbox(self, mate_func):
        toolbox = base.Toolbox()
        if mate_func.__name__ == 'cxOnePoint':
            toolbox.register("mate", mate_func)
        else:
            toolbox.register("mate", mate_func, termpb=0.5)
        toolbox.register("mutate", gp.mutInsert, pset=self.pset)
        toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("print_name", self.print_name)

        return toolbox

    def evolve_population(self, pop, NGEN):
        CXPB, MUTPB = 0.5, 0.2

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return pop

    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.pop = self.evolve_population(self.pop, 100)
        self.toolbox.print_name()
        print(self.toolbox.mate)
