from creamas.core.artifact import Artifact

import numpy as np
from deap import tools
from deap import creator
from deap import base
import deap.gp as gp


class GeneticImageArtifact(Artifact):
    def __init__(self, creator, obj, function_tree):
        super().__init__(creator, obj, domain='image')
        self._function_tree = function_tree

    @property
    def function_tree(self):
        return self._function_tree

    @property
    def pset(self):
        return self._pset

    @staticmethod
    def max_distance(artifact):
        return np.sqrt(artifact.obj.shape[0])

    @staticmethod
    def distance(artifact1, artifact2):
        return np.sqrt(np.sum(np.square(artifact1 - artifact2)))

    @staticmethod
    def generate_image(individual, width=32, height=32):
        func = gp.compile(individual, individual.pset)
        image = np.zeros((width, height))

        coords = [(x, y) for x in range(width) for y in range(height)]
        for coord in coords:
            x = coord[0]
            y = coord[1]
            x_normalized = x / width - 0.5
            y_normalized = y / height - 0.5
            color_value = np.abs(func(x_normalized, y_normalized)) * 255
            image[x, y] = np.around(color_value)

            if image[x, y] < 0:
                image[x, y] = 0
            elif image[x, y] > 255:
                image[x, y] = 255

        return image

    @staticmethod
    def evaluate(individual, agent):
        if individual.image is None:
            image = GeneticImageArtifact.generate_image(individual)
            individual.image = image
        artifact = GeneticImageArtifact(agent, individual.image, individual)
        evaluation, _ = agent.evaluate(artifact)
        return evaluation,

    @staticmethod
    def evolve_population(population, generations, toolbox):
        CXPB, MUTPB = 0.5, 0.2

        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for g in range(generations):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    del child1.image
                    del child2.image

            for mutant in offspring:
                if np.random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring

    @staticmethod
    def create(generations, agent, toolbox, pset):
        population = GeneticImageArtifact.create_population(pset)
        toolbox.register("evaluate", GeneticImageArtifact.evaluate, agent=agent)
        GeneticImageArtifact.evolve_population(population, generations, toolbox)
        best = tools.selBest(population, 1)[0]

        return best

    @staticmethod
    def create_population(pset, size=10):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                       pset=pset, image=None)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox.population(size)

    @staticmethod
    def invent(n, agent, create_kwargs):
        # pop_size = 10
        # if len(agent.stmem.artifacts) < pop_size:
        #     pop_size = len(agent.stmem.artifacts)
        # mem_arts = np.random.choice(agent.stmem.artifacts, size=pop_size, replace=False)
        # population = []
        # for art in mem_arts:
        #     population.append(art.function_tree)
        function_tree = GeneticImageArtifact.create(n, agent, **create_kwargs)
        artifact = GeneticImageArtifact(agent, function_tree.image, function_tree)
        return artifact
