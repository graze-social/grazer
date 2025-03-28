import numpy as np
from app.algos.base import BaseParser


class LogicParser(BaseParser):
    async def param_compare(self, records, input_value, operator, value):
        return np.array(
            LogicEvaluator.compare([input_value for e in records], operator, value),
            dtype=bool,
        )

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation("param_compare", self.param_compare)
