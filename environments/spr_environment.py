from creamas.mp import MultiEnvironment
import asyncio
import aiomas

class SprEnvironment(MultiEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_agent_acquaintances(self):
        agents = self.get_agents(address=False)
        addresses = self.get_agents(address=True)

        for agent in agents:
            aiomas.run(until=agent.set_acquaintances(addresses))

    def get_total_reward(self):
        agents = self.get_agents(address=False)

        total_reward = 0

        for agent in agents:
            total_reward += aiomas.run(until=agent.get_total_reward())

        return total_reward

    def log_situation(self, step):
        agents = self.get_agents(address=False)

        for agent in agents:
            aiomas.run(until=agent.log_situation())

