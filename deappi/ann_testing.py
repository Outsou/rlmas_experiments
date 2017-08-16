from experiments.feature.util import create_pset
from deappi.kokkeilu import generate_color_image, evolve_population

from deap import base, creator, gp, tools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import time
import sys
import urllib
import tarfile


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                _temp_path, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                _temp_path, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

    def string_to_id(self, name):
        ids = []
        for id, string in self.node_lookup.items():
            names = list(map(lambda s: s.strip(), string.split(',')))
            if name in names:
                ids.append(id)
        return ids

_temp_path = '/tmp/imagenet'
_node_lookup = None
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def class_probability(image, classname):
    predictions = run_inference_on_image(image)
    class_ids = _node_lookup.string_to_id(classname)
    return predictions[class_ids[0]]


def evaluate(individual):
    image = generate_color_image(individual, 128, 128)
    if image is None:
        return -1,
    return class_probability(image, 'banana'),


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
           _temp_path, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg:0': image})
        predictions = np.squeeze(predictions)

        return predictions

def mutate(individual, pset, expr):
    rand = np.random.rand()
    if rand <= 0.25:
        return gp.mutShrink(individual),
    elif rand <= 0.5:
        return gp.mutInsert(individual, pset)
    elif rand <= 0.75:
        return gp.mutNodeReplacement(individual, pset)
    return gp.mutUniform(individual, expr, pset)

def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = _temp_path
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
              filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


if __name__ == '__main__':
    maybe_download_and_extract()
    _node_lookup = NodeLookup()

    start_time = time.time()

    pset = create_pset()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                   pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)
    toolbox.register("evaluate", evaluate)

    create_graph()

    pop = toolbox.population(n=10)

    pop, best_in_gen = evolve_population(pop, 200, toolbox)

    print('Run time: {0:.2f}s'.format(time.time() - start_time))
    start_time = time.time()

    i = 0
    folder = 'gen_best'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    for best in best_in_gen:
        i += 1
        img = generate_color_image(best, 128, 128)
        plt.imshow(img)
        plt.title('Eval: {}'.format(best.fitness.values[0]))
        plt.savefig('{}/gen{}'.format(folder, i))

    print('Save time: {0:.2f}s'.format(time.time() - start_time))

    top_best = sorted(range(len(best_in_gen)), key=lambda x: best_in_gen[x].fitness.values[0], reverse=True)[:20]
    top_best = [x+1 for x in top_best]
    print(top_best)

    # best = tools.selBest(pop, 1)[0]
    # eval = evaluate(best)
    # print(eval)
    # img = generate_color_image(best, 128, 128)
    # plt.imshow(img)
    # plt.show()

    # img_path = os.path.join(_temp_path, 'cropped_panda.jpg')
    # import scipy.ndimage as nd
    # img = nd.imread(img_path)
    #
    # predictions = run_inference_on_image(img)
    # top_k = predictions.argsort()[-5:][::-1]
    # for node_id in top_k:
    #     human_string = _node_lookup.id_to_string(node_id)
    #     score = predictions[node_id]
    #     print('%s (score = %.5f)' % (human_string, score))
    #
    # panda_ids = _node_lookup.string_to_id('panda')
    # for id in panda_ids:
    #     print()
    #     print(id)
    #     print(_node_lookup.id_to_string(id))
    #     print(predictions[id])



