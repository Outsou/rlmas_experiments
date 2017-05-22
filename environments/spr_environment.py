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

