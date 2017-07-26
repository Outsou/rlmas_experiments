from creamas.core.artifact import Artifact

import numpy as np
from deap import tools
from deap import creator
from deap import base
import deap.gp as gp

import cv2


class GeneticImageArtifact(Artifact):
    def __init__(self, creator, obj, function_tree):
        super().__init__(creator, obj, domain='image')
        self.framings['function_tree'] = function_tree

    @staticmethod
    def max_distance(create_kwargs):
        shape = create_kwargs['shape']
        return np.sqrt(np.sum(np.square(np.ones(shape))))

    @staticmethod
    def distance(artifact1, artifact2):
        img1 = cv2.cvtColor(artifact1.obj, cv2.COLOR_RGB2GRAY) / 255
        img2 = cv2.cvtColor(artifact2.obj, cv2.COLOR_RGB2GRAY) / 255
        return np.sqrt(np.sum(np.square(img1 - img2)))

    @staticmethod
    def generate_image(func, shape=(32, 32)):
        width = shape[0]
        height = shape[1]
        image = np.zeros((width, height, 3))

        coords = [(x, y) for x in range(width) for y in range(height)]
        for coord in coords:
            x = coord[0]
            y = coord[1]
            x_normalized = x / width - 0.5
            y_normalized = y / height - 0.5
            color_value = np.around(np.array(func(x_normalized, y_normalized)))
            for i in range(3):
                if color_value[i] < 0:
                    image[x, y, i] = 0
                elif color_value[i] > 255:
                    image[x, y, i] = 255
                else:
                    image[x, y, i] = color_value[i]

        return np.uint8(image)

    @staticmethod
    def evaluate(individual, agent, shape):
        if individual.image is None:
            # If tree is too tall return negative evaluation
            try:
                func = gp.compile(individual, individual.pset)
            except MemoryError:
                return -1,
            image = GeneticImageArtifact.generate_image(func, shape)
            individual.image = image
        artifact = GeneticImageArtifact(agent, individual.image, individual)
        evaluation, _ = agent.evaluate(artifact)
        return evaluation,

    @staticmethod
    def evolve_population(population, generations, toolbox, pset):
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
                    toolbox.mutate(mutant, pset)
                    del mutant.fitness.values
                    if mutant.image is not None:
                        del mutant.image

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring

    @staticmethod
    def create(generations, agent, toolbox, pset, pop_size, shape):
        population = []

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                       pset=pset, image=None)

        if len(agent.stmem.artifacts) > 0:
            mem_size = min(pop_size, len(agent.stmem.artifacts))
            mem_arts = np.random.choice(agent.stmem.artifacts, size=mem_size, replace=False)
            for art in mem_arts:
                individual = creator.Individual(art.framings['function_tree'])
                population.append(individual)

        if len(population) < pop_size:
            pop_toolbox = base.Toolbox()
            pop_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
            pop_toolbox.register("individual", tools.initIterate, creator.Individual,
                                 pop_toolbox.expr)
            pop_toolbox.register("population", tools.initRepeat, list, pop_toolbox.individual)
            population += pop_toolbox.population(pop_size - len(population))

        toolbox.register("evaluate", GeneticImageArtifact.evaluate, agent=agent, shape=shape)
        GeneticImageArtifact.evolve_population(population, generations, toolbox, pset)
        best = tools.selBest(population, 1)[0]
        return best

    @staticmethod
    def invent(n, agent, create_kwargs):
        function_tree = GeneticImageArtifact.create(n, agent, **create_kwargs)
        artifact = GeneticImageArtifact(agent, function_tree.image, list(function_tree))
        return artifact, None
