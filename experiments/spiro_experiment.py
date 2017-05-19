from environments.spr_environment import SprEnvironment
from agents.spr_agent import SprAgent
import creamas.examples.spiro.spiro

import scipy.misc


if __name__ == "__main__":
    env = SprEnvironment.create(('localhost', 5555))

    # art = SprAgent(env, ((120, 120))).create(50, 100)
    # scipy.misc.imsave('test.jpg', art)
    # art2 = SprAgent(env, ((120, 120))).create(50, -100)
    # scipy.misc.imsave('test2.jpg', art2)

    SprAgent(env, ((25, 25), (85, 85)))
    SprAgent(env, ((25, -25), (85, -85)))
    SprAgent(env, ((-25, 25), (-85, 85)))

    env.set_agent_acquaintances()