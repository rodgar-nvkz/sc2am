"""A simple StarCraft II bot that builds workers and expands."""

from sc2.bot_ai import BotAI


class SimpleBot(BotAI):
    async def on_step(self, iteration: int):
        # After 50 iterations (~3.5 seconds), send some workers to attack
        if iteration == 50:
            for worker in self.workers.random_group_of(5):
                if self.enemy_start_locations:
                    worker.attack(self.enemy_start_locations[0])
