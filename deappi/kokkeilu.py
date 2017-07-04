if __name__ == '__main__':

    import operator
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pylab as pl
    import cv2

    from deap import base
    from deap import creator
    from deap import gp
    from deap import tools

    np.seterr(all='raise')

    # Make the tree

    def divide(a, b):
        if b == 0:
            b = 0.000001
        return np.divide(a, b)

    def log(a):
        if a <= 0:
            a = 0.000001
        return np.log(a)

    def exp(a):
        if a > 100:
            a = 100
        elif a < -100:
            a = -100
        return np.exp(a)

    def generate_image(individual, width, height):
        func = gp.compile(individual, individual.pset)
        image = np.zeros((width, height))

        coords = [(x, y) for x in range(width) for y in range(height)]
        for coord in coords:
            x = coord[0]
            y = coord[1]
            x_normalized = x / width - 0.5
            y_normalized = y / height - 0.5
            color_value = np.abs(func(x_normalized, y_normalized)) * 255
            #print(color_value)
            image[x, y] = np.around(color_value)

            if image[x, y] < 0:
                image[x, y] = 0
            elif image[x, y] > 255:
                image[x, y] = 255

        # image_ = image / 255
        # my_cmap = plt.cm.get_cmap('rainbow')
        # color_array = my_cmap(image_)

        return image

    def evaluate(individual):
        image = generate_image(individual, 32, 32)
        # image_ = image / 255
        # my_cmap = plt.cm.get_cmap('rainbow')
        # color_array = my_cmap(image_)
        # evaluation = np.sum(color_array[:, :, 1])

        edges = get_edges(image)
        evaluation = box_count(edges)
        # evaluation = -abs(1.8 - evaluation)

        # evaluation = 0
        # for i in range(32):
        #     for j in range(32):
        #         if image[i, j] > 0 and image[i, j] < 255:
        #             evaluation += 1

        return evaluation,

    def draw_tree(tree):
        nodes, edges, labels = gp.graph(tree)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.nx_pydot.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()


    def box_count(image):
        pixels = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] > 0:
                    pixels.append((i, j))
        lx = image.shape[1]
        ly = image.shape[0]
        pixels = pl.array(pixels)
        if len(pixels) < 2:
            return 0
        scales = np.logspace(1, 4, num=20, endpoint=False, base=2)
        Ns = []
        for scale in scales:
            # print('Scale: ' + str(scale))
            H, edges = np.histogramdd(pixels, bins=(np.arange(0, lx, scale), np.arange(0, ly, scale)))
            H_sum = np.sum(H > 0)
            if H_sum == 0:
                H_sum = 1
            Ns.append(H_sum)

        coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
        # asd = zip(scales, Ns)
        # for thing in asd:
        #     print(thing)
        hausdorff_dim = -coeffs[0]
        # print('The Hausdorff dimension is: ' + str(hausdorff_dim))
        return hausdorff_dim


    def show_individual(individual):
        img = generate_image(individual, 32, 32)
        plt.imshow(img, cmap='gray')
        plt.show()

    def get_edges(img):
        img_uint8 = np.uint8(img)
        edges = cv2.Canny(img_uint8, 100, 200)
        return edges

    pset = gp.PrimitiveSet("MAIN", arity=2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    # pset.addPrimitive(max, 2)
    # pset.addPrimitive(min, 2)
    # pset.addPrimitive(divide, 2)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.tan, 1)
    #pset.addPrimitive(np.arcsin, 1)
    #pset.addPrimitive(np.arccos, 1)
    #pset.addPrimitive(np.arctan, 1)
    pset.addPrimitive(exp, 1)
    #pset.addPrimitive(np.sqrt, 1)
    pset.addPrimitive(log, 1)
    pset.addEphemeralConstant('rand', lambda: np.random.randint(1, 4))

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")


    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                   pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutInsert, pset=pset)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)
    toolbox.register("evaluate", evaluate)

    # tree = toolbox.individual()
    # img = generate_image(tree, 32, 32)
    # plt.imshow(img)
    # plt.show()
    #print(evaluate(tree))

    def evolve_population(pop, NGEN):
        CXPB, MUTPB = 0.5, 0.2

        # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

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
            pop[:] = offspring

        return pop

    pop = toolbox.population(n=50)

    pop = evolve_population(pop, 50)

    best = tools.selBest(pop, 1)[0]
    eval = evaluate(best)
    print(eval)
    img = generate_image(best, 128, 128)
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(get_edges(img), cmap='gray')
    plt.show()

    # while True:
    #     ind = toolbox.individual()
    #     print(evaluate(ind))
    #     img = generate_image(ind, 32, 32)
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    #     edges = get_edges(img)
    #     print(box_count(edges))
    #     plt.imshow(edges, cmap='gray')
    #     plt.show()
    #     print('...')

