from creamas.core.environment import Environment

class SprEnvironment(Environment):

    def set_agent_acquaintances(self):
        agents = self.get_agents(address=False)

        for agent in agents:
            agent.set_acquaintances()