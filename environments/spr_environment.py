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

    def get_acquaintance_counts(self):
        agents = self.get_agents(address=False)

        acquaintances = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            acquaintances[name] = aiomas.run(until=agent.get_acquaintances())

        return acquaintances

    def get_acquaintance_values(self):
        agents = self.get_agents(address=False)

        acquaintance_values = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            acquaintance_values[name] = aiomas.run(until=agent.get_acquaintance_values())

        return acquaintance_values

    def get_comparison_count(self):
        agents = self.get_agents(address=False)

        total_comparisons = 0

        for agent in agents:
            total_comparisons += aiomas.run(until=agent.get_comparison_count())

        return total_comparisons