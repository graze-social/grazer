import numpy as np
from app.helpers import is_truthy, is_list_of_lists


class LogicEvaluator:
    def __init__(self):
        self.operations = {}
        self.gpu_accelerable_operations = set()
        self.gpu_accelerable_custom_operations = set()

    @classmethod
    async def initialize(cls):
        """Async factory method to handle async initialization."""
        return cls()

    async def add_operation(
        self, name, func, gpu_accelerable=False, gpu_accelerable_custom=False
    ):
        """Registers a custom operation."""
        self.operations[name] = func
        if gpu_accelerable:
            self.gpu_accelerable_operations.add(name)
        if gpu_accelerable_custom:
            self.gpu_accelerable_custom_operations.add(name)

    async def evaluate_with_audit(self, cond, records):
        """
        Evaluates a JSON-like condition structure across all records without short-circuiting
        and accumulates the results for auditing in a structure that mirrors the input.

        Args:
            cond (dict): The condition to evaluate.
            records (list): The list of records to evaluate.

        Returns:
            dict: A dictionary with 'results' (list of booleans per record)
                  and 'audit' (structured details of evaluations that mirror the input condition).
        """
        if not records:
            return {"results": [], "audit": {}}

        def record_audit(condition, results, scores, sub_audit=None):
            """
            Helper to record the evaluation path, results, and condition for auditing.
            Converts numpy booleans to Python booleans and structures audit data.
            """
            audit_entry = {
                "condition": condition,
                "scores": scores,
                "result": [
                    bool(r) for r in results
                ],  # Ensure results are Python booleans
            }
            if sub_audit:
                audit_entry["sub_audit"] = sub_audit
            return audit_entry

        if isinstance(cond, dict) and "and" in cond:
            sub_audits = []
            results = [True] * len(
                records
            )  # Start assuming all records satisfy the condition
            for sub_condition in cond["and"]:
                sub_evaluation = await self.evaluate_with_audit(sub_condition, records)
                sub_audits.append(sub_evaluation["audit"])
                # Aggregate results for `and`: all sub-results must be True
                results = [
                    res and sub_res
                    for res, sub_res in zip(results, sub_evaluation["results"])
                ]

            return {
                "results": [bool(r) for r in results],
                "audit": record_audit(cond, results, [], sub_audits),
            }

        elif isinstance(cond, dict) and "or" in cond:
            sub_audits = []
            results = [False] * len(
                records
            )  # Start assuming no records satisfy the condition
            for sub_condition in cond["or"]:
                sub_evaluation = await self.evaluate_with_audit(sub_condition, records)
                sub_audits.append(sub_evaluation["audit"])
                # Aggregate results for `or`: at least one sub-result must be True
                results = [
                    res or sub_res
                    for res, sub_res in zip(results, sub_evaluation["results"])
                ]

            return {
                "results": [bool(r) for r in results],
                "audit": record_audit(cond, results, [], sub_audits),
            }

        else:
            # Leaf-level condition
            indices = list(range(len(records)))
            leaf_results = await self._evaluate_condition(cond, records, indices)
            scores = await self._evaluate_scores(cond, records, indices)
            # Convert numpy booleans to Python booleans for serialization
            leaf_results = [bool(r) for r in leaf_results]
            return {
                "results": leaf_results,
                "audit": record_audit(cond, leaf_results, scores),
            }

    async def evaluate(self, cond, records):
        """
        Evaluates a JSON-like condition structure across a batch of records.
        Handles logical operators (`and`, `or`) and ensures correct mapping of results.
        """
        if not records:
            return []

        # Start evaluation with all indices active
        active_indices = list(range(len(records)))

        # Dispatch based on condition type
        if isinstance(cond, dict) and "and" in cond:
            # Logical AND: All sub-conditions must be True
            for sub_condition in cond["and"]:
                sub_results = await self.evaluate(
                    sub_condition, [records[i] for i in active_indices]
                )
                active_indices = [
                    i for i, res in zip(active_indices, sub_results) if is_truthy(res)
                ]
                if not active_indices:
                    # Short-circuit: No records satisfy this condition
                    return [False] * len(records)

            # Map results back to original indices
            results = [False] * len(records)
            for idx in active_indices:
                results[idx] = True
            return results

        elif isinstance(cond, dict) and "or" in cond:
            # Logical OR: At least one sub-condition must be True
            results = [False] * len(records)
            for sub_condition in cond["or"]:
                inactive_indices = [i for i in range(len(records)) if not results[i]]
                if not inactive_indices:
                    # Short-circuit: All records already satisfy this condition
                    return results

                sub_results = await self.evaluate(
                    sub_condition, [records[i] for i in inactive_indices]
                )
                for idx, res in zip(inactive_indices, sub_results):
                    if is_truthy(res):
                        results[idx] = True

            return results

        else:
            # Leaf-level condition
            return await self._evaluate_condition(cond, records, active_indices)

    async def _evaluate_scores(self, cond, records, indices):
        if not indices:
            return []
    
        sub_records = [records[i] for i in indices]
    
        for op, params in cond.items():
            if op in self.operations:
                operation = self.operations[op]
                if hasattr(operation, "__self__") and hasattr(operation.__self__, "get_ml_scores"):
                    sub_results = await operation.__self__.get_ml_scores(sub_records, *params)
                else:
                    sub_results = await operation(sub_records, *params)
                    sub_results = np.array(sub_results).astype(float)

                results = [0.0] * len(records)
                for idx, res in zip(indices, sub_results):
                    results[idx] = res
                return results
        raise ValueError(f"Unknown operation '{op}'")

    async def _evaluate_condition(self, cond, records, indices):
        """
        Evaluates a single condition on a subset of records specified by indices.
        Ensures that the condition is a leaf-level operation.
        """
        if not indices:
            return []

        sub_records = [records[i] for i in indices]
        # Leaf-level condition
        for op, params in cond.items():
            if op in self.operations:
                sub_results = await self.operations[op](sub_records, *params)
                results = [False] * len(records)
                for idx, res in zip(indices, sub_results):
                    results[idx] = res
                return results
            else:
                raise ValueError(f"Unknown operation '{op}'")

        raise ValueError("Invalid condition structure.")

    @staticmethod
    async def compare(value, operator, threshold):
        """Performs comparison based on the specified operator."""
        if operator == "==":
            if is_list_of_lists(value):
                return check_empty_string(value, threshold)
            else:
                return value == threshold
        elif operator == ">=":
            if threshold == None:
                return value == threshold
            else:
                return value >= threshold
        elif operator == "<=":
            if threshold == None:
                return value == threshold
            else:
                return value <= threshold
        elif operator == ">":
            if threshold == None:
                return value == threshold
            else:
                return value > threshold
        elif operator == "<":
            if threshold == None:
                return value == threshold
            else:
                return value < threshold
        elif operator == "!=":
            if is_list_of_lists(value):
                return ~check_empty_string(value, threshold)
            else:
                return value != threshold
        elif operator == "in":
            return np.isin(value, threshold)
        elif operator == "not_in":
            return ~np.isin(value, threshold)
        else:
            raise ValueError(f"Unknown comparator '{operator}'")

    @staticmethod
    async def extract_conditions(condition):
        """
        Extracts all operator definitions within `and` or `or` conditions.
        :param condition: JSON-like condition structure.
        :return: A flattened list of operator definitions.
        """
        extracted = []

        def traverse(cond):
            if isinstance(cond, dict):
                for key, value in cond.items():
                    if key in {"and", "or"}:
                        for sub_cond in value:
                            traverse(sub_cond)
                    else:
                        # Assume any other key is an operator
                        extracted.append({key: value})
            elif isinstance(cond, list):
                for item in cond:
                    traverse(item)

        traverse(condition)
        return extracted

    def sort_conditions(self, condition):
        """
        Sorts conditions under 'and'/'or' keys in the structure based on
        whether their keys are in `self.gpu_accelerable_operations`.
        Args:
            condition (dict): The condition structure to sort.
        Returns:
            dict: The sorted condition structure.
        """
        if isinstance(condition, dict):
            # Check if 'and' or 'or' keys are present
            if "and" in condition:
                condition["and"] = sorted(
                    condition["and"],
                    key=lambda cond: any(
                        k in self.gpu_accelerable_operations for k in cond
                    ),
                )
                # Recursively process each sub-condition
                condition["and"] = [
                    self.sort_conditions(sub_cond) for sub_cond in condition["and"]
                ]
            elif "or" in condition:
                condition["or"] = sorted(
                    condition["or"],
                    key=lambda cond: any(
                        k in self.gpu_accelerable_operations for k in cond
                    ),
                )
                # Recursively process each sub-condition
                condition["or"] = [
                    self.sort_conditions(sub_cond) for sub_cond in condition["or"]
                ]
            else:
                # Process leaf-level conditions
                for key, value in condition.items():
                    if isinstance(value, (list, dict)):
                        condition[key] = self.sort_conditions(value)
        elif isinstance(condition, list):
            # Process lists recursively
            condition = [self.sort_conditions(item) for item in condition]
        return condition

    @staticmethod
    def rehydrate_single_manifest(manifest, condition_map):
        """
        Rehydrates a single manifest by replacing IDs in the manifest with
        their corresponding conditions from the condition map.

        Args:
            manifest (dict): The single algorithm manifest to process.
            condition_map (dict): A dictionary mapping condition IDs to condition parameters.

        Returns:
            dict: The rehydrated manifest with condition parameters.
        """
        def rehydrate_condition(condition):
            """
            Recursively replaces IDs with condition parameters.
            """
            if isinstance(condition, int):
                # Replace the integer ID with its corresponding condition
                condition_data = condition_map.get(condition)
                if condition_data and condition_data.get("algorithm_component_id"):
                    result = condition_data["condition_parameters"]
                else:
                    if not condition_data:
                        raise ValueError(f"Condition ID {condition} not found in condition_map.")
                    operator_name = condition_data.get("operator_name")
                    result = {operator_name: condition_data["condition_parameters"]}
                return result
            elif isinstance(condition, dict):
                # Rehydrate nested conditions in a dictionary
                return {
                    key: [rehydrate_condition(sub_cond) for sub_cond in value]
                    for key, value in condition.items()
                    if key in {"and", "or"}  # Only process `and`/`or` keys
                }
            elif isinstance(condition, list):
                # Rehydrate a list of conditions
                return [rehydrate_condition(sub_cond) for sub_cond in condition]
            else:
                raise ValueError(f"Unexpected condition format: {condition}")
        # Rehydrate the manifest
        if manifest:
            rehydrated_manifest = {
                key: rehydrate_condition(value) if key == "filter" else value
                for key, value in manifest.items()
            }
            return rehydrated_manifest
        return {}
