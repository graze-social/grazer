import numpy as np
from app.algos.base import BaseParser
from app.logic_evaluator import LogicEvaluator


class SocialParser(BaseParser):
    def temp_hack(self, asset):
        if isinstance(asset, dict) and asset.get("asset_value"):
            asset = np.array(asset.get("asset_value"))
        return asset

    async def get_starter_pack(self, starter_pack_url):
        return await self.network_worker.get_starter_pack.remote(starter_pack_url)

    async def get_list(self, list_url):
        return await self.network_worker.get_list.remote(list_url)

    async def get_user_collection(self, actor_handle, direction):
        return await self.network_worker.get_user_collection.remote(
            actor_handle, direction
        )

    async def get_magic_audience(self, audience_id):
        return await self.network_worker.get_magic_audience.remote(audience_id)

    async def check_memberships(self, records, operator, comparison_set):
        records_array = np.array([record["did"] for record in records])
        return await LogicEvaluator.compare(records_array, operator, comparison_set)

    async def starter_pack_member(self, records, starter_pack_url, operator):
        """Resolve the attribute path and apply a comparison operation using LogicEvaluator.compare."""
        asset = await self.get_starter_pack(starter_pack_url)
        return await self.check_memberships(records, operator, self.temp_hack(asset))

    async def list_member(self, records, list_url, operator):
        """Resolve the attribute path and apply a comparison operation using LogicEvaluator.compare."""
        asset = await self.get_list(list_url)
        return await self.check_memberships(records, operator, self.temp_hack(asset))

    async def social_graph(self, records, actor_handle, operator, direction):
        """Resolve the attribute path and apply a comparison operation using LogicEvaluator.compare."""
        asset = await self.get_user_collection(actor_handle, direction)
        return await self.check_memberships(records, operator, self.temp_hack(asset))

    async def social_list(self, records, actor_did_list, operator):
        """Resolve the attribute path and apply a comparison operation using LogicEvaluator.compare."""
        outdids = []
        for actor in actor_did_list:
            outdids.append(await self.network_worker.get_or_set_handle_did.remote(actor))
        return await self.check_memberships(records, operator, np.array(outdids))

    async def magic_audience(self, records, audience_id, operator):
        """Resolve the attribute path and apply a comparison operation using LogicEvaluator.compare."""
        asset = await self.get_magic_audience(audience_id)
        return await self.check_memberships(records, operator, self.temp_hack(asset))

    async def register_operations(self, logic_evaluator):
        # Register attribute comparison operation
        await logic_evaluator.add_operation("social_graph", self.social_graph)
        await logic_evaluator.add_operation("social_list", self.social_list)
        await logic_evaluator.add_operation(
            "starter_pack_member", self.starter_pack_member
        )
        await logic_evaluator.add_operation("list_member", self.list_member)
        await logic_evaluator.add_operation("magic_audience", self.magic_audience)
