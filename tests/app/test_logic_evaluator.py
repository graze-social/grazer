import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from app.logic_evaluator import LogicEvaluator


import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from app.logic_evaluator import LogicEvaluator

@pytest_asyncio.fixture
async def evaluator():
    """
    Create a fresh LogicEvaluator for each test,
    and register mock operations to ensure coverage
    of leaf-level conditions.
    """
    ev = await LogicEvaluator.initialize()

    async def mock_op(records, *args):
        return [rec.get("value") in args for rec in records]

    async def mock_op_get_ml_scores(records, *args):
        return np.array([0.5] * len(records))

    await ev.add_operation("mock_op", mock_op, gpu_accelerable=True)

    class MockModel:
        async def get_ml_scores(self, records, *args):
            return await mock_op_get_ml_scores(records, *args)

    mock_model_instance = MockModel()

    async def mock_scores_op(records, *args):
        return np.array([0.0] * len(records))  # Ensures boolean evaluation returns False

    mock_scores_op.__self__ = mock_model_instance
    await ev.add_operation("mock_scores_op", mock_scores_op, gpu_accelerable=True, gpu_accelerable_custom=True)

    return ev


@pytest.mark.asyncio
async def test_initialize():
    ev = await LogicEvaluator.initialize()
    assert isinstance(ev, LogicEvaluator)
    assert ev.operations == {}
    assert ev.gpu_accelerable_operations == set()
    assert ev.gpu_accelerable_custom_operations == set()


@pytest.mark.asyncio
async def test_add_operation(evaluator):
    # The fixture already added two operations. Check them:
    assert "mock_op" in evaluator.operations
    assert evaluator.gpu_accelerable_operations == {"mock_op", "mock_scores_op"}
    assert evaluator.gpu_accelerable_custom_operations == {"mock_scores_op"}


@pytest.mark.asyncio
async def test_evaluate_leaf_condition(evaluator):
    records = [
        {"value": 1},
        {"value": 2},
        {"value": 3},
    ]
    # Condition is: "mock_op": [2, 3]
    # i.e. True if record["value"] is either 2 or 3
    cond = {"mock_op": [2, 3]}
    results = await evaluator.evaluate(cond, records)
    # Expect [False, True, True]
    assert results == [False, True, True]


@pytest.mark.asyncio
async def test_evaluate_and(evaluator):
    # Condition: and => [ {"mock_op": [2, 3]}, {"mock_op": [3]} ]
    # This is True only if record["value"] is in [2, 3] *and* in [3].
    records = [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 3}]
    cond = {"and": [{"mock_op": [2, 3]}, {"mock_op": [3]}]}
    results = await evaluator.evaluate(cond, records)
    # Step 1: "mock_op": [2, 3] => [False, True, True, True]
    # Step 2: "mock_op": [3] => for the subset that was True => records at indices 1, 2, 3 -> [False, True, True]
    # So overall => [False, False, True, True]
    assert results == [False, False, True, True]


@pytest.mark.asyncio
async def test_evaluate_and_short_circuit(evaluator):
    """
    If the first sub-condition reduces active_indices to empty,
    we short-circuit and return [False, False, False, ...].
    """
    records = [{"value": 1}, {"value": 2}]
    # First sub-condition returns all False => active_indices becomes empty
    cond = {"and": [{"mock_op": [999]}, {"mock_op": [2]}]}
    results = await evaluator.evaluate(cond, records)
    # first sub-condition => [False, False], so short-circuit
    assert results == [False, False]


@pytest.mark.asyncio
async def test_evaluate_or(evaluator):
    # Condition: or => [ {"mock_op": [1]}, {"mock_op": [3]} ]
    # True if record["value"] is 1 or 3
    records = [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 3}]
    cond = {"or": [{"mock_op": [1]}, {"mock_op": [3]}]}
    results = await evaluator.evaluate(cond, records)
    # "mock_op": [1] => [True, False, False, False]
    # "mock_op": [3] => for the records still False => indices 1, 2, 3 => [False, True, True]
    # => overall [True, False OR False, False OR True, False OR True] => [True, False, True, True]
    assert results == [True, False, True, True]


@pytest.mark.asyncio
async def test_evaluate_or_short_circuit(evaluator):
    """
    If an OR condition has turned all records to True early,
    we short-circuit the rest of sub-conditions.
    """
    records = [{"value": 1}, {"value": 2}]
    # The first sub-condition => [True, True], then short-circuits
    cond = {"or": [{"mock_op": [1, 2]}, {"mock_op": [999]}]}
    results = await evaluator.evaluate(cond, records)
    # first sub-condition => [True, True], short-circuit => done
    assert results == [True, True]


@pytest.mark.asyncio
async def test_evaluate_empty_records(evaluator):
    cond = {"mock_op": [1, 2]}
    # Evaluate with empty record list => return []
    results = await evaluator.evaluate(cond, [])
    assert results == []


@pytest.mark.asyncio
async def test_evaluate_unknown_operation(evaluator):
    records = [{"value": 1}]
    cond = {"unknown_op": [1]}
    with pytest.raises(ValueError, match="Unknown operation 'unknown_op'"):
        await evaluator.evaluate(cond, records)


@pytest.mark.asyncio
async def test_evaluate_with_audit_leaf(evaluator):
    records = [{"value": 1}, {"value": 2}, {"value": 3}]
    cond = {"mock_op": [2]}
    audit_result = await evaluator.evaluate_with_audit(cond, records)
    # results => [False, True, False]
    assert audit_result["results"] == [False, True, False]
    # "audit" should mirror the leaf condition
    assert "condition" in audit_result["audit"]
    assert "scores" in audit_result["audit"]
    assert audit_result["audit"]["result"] == [False, True, False]


@pytest.mark.asyncio
async def test_evaluate_with_audit_and(evaluator):
    records = [{"value": 1}, {"value": 2}, {"value": 3}]
    cond = {"and": [{"mock_op": [2, 3]}, {"mock_op": [3]}]}
    audit_result = await evaluator.evaluate_with_audit(cond, records)
    # We'll get [False, False, True]
    assert audit_result["results"] == [False, False, True]
    # There's a "sub_audit" entry for each sub-condition
    sub_audit = audit_result["audit"]["sub_audit"]
    assert len(sub_audit) == 2  # the two subconditions
    # Confirm the final "result" matches the overall results
    assert audit_result["audit"]["result"] == [False, False, True]


@pytest.mark.asyncio
async def test_evaluate_with_audit_empty_records(evaluator):
    cond = {"mock_op": [1, 2]}
    result = await evaluator.evaluate_with_audit(cond, [])
    assert result == {"results": [], "audit": {}}


@pytest.mark.asyncio
async def test_evaluate_with_audit_scores(evaluator):
    """
    Test that _evaluate_scores is used if the operation has 'get_ml_scores'.
    We'll use 'mock_scores_op' which uses a mock model that returns 0.5.
    """
    records = [{"value": 123}, {"value": 456}]
    cond = {"mock_scores_op": [1, 2]}
    result = await evaluator.evaluate_with_audit(cond, records)
    assert result["results"] == [False, False]  # from the real boolean evaluation
    # But "scores" should be set to the array of 0.5
    assert result["audit"]["scores"] == [0.5, 0.5]


@pytest.mark.asyncio
async def test_evaluate_with_audit_unknown_operator(evaluator):
    records = [{"value": 1}]
    cond = {"unknown_op": [1]}
    with pytest.raises(ValueError, match="Unknown operation 'unknown_op'"):
        await evaluator.evaluate_with_audit(cond, records)


@pytest.mark.asyncio
async def test_compare():
    """
    Test the static compare method with all supported operators,
    including threshold=None cases.
    """
    # operator == 
    assert await LogicEvaluator.compare(5, "==", 5) is True
    assert await LogicEvaluator.compare(5, "==", 4) is False
    # operator !=
    assert await LogicEvaluator.compare(5, "!=", 4) is True
    assert await LogicEvaluator.compare(5, "!=", 5) is False
    # operator >=
    assert await LogicEvaluator.compare(5, ">=", 5) is True
    assert await LogicEvaluator.compare(4, ">=", None) is False  # special case
    # operator <=
    assert await LogicEvaluator.compare(5, "<=", 5) is True
    assert await LogicEvaluator.compare(5, "<=", None) is False  # special case
    # operator >
    assert await LogicEvaluator.compare(6, ">", 5) is True
    assert await LogicEvaluator.compare(5, ">", 5) is False
    assert await LogicEvaluator.compare(5, ">", None) is False
    # operator <
    assert await LogicEvaluator.compare(4, "<", 5) is True
    assert await LogicEvaluator.compare(5, "<", None) is False
    # operator in
    assert all(await LogicEvaluator.compare(np.array([1,2]), "in", [1,2,3]))
    # operator not_in
    assert not any(await LogicEvaluator.compare(np.array([1,2]), "not_in", [1,2,3]))
    # unknown comparator
    with pytest.raises(ValueError, match="Unknown comparator 'abcd'"):
        await LogicEvaluator.compare(5, "abcd", 5)


@pytest.mark.asyncio
async def test_extract_conditions():
    # Condition with nested and/or
    condition = {
        "and": [
            {"mock_op": [1]},
            {
                "or": [
                    {"mock_op": [2]},
                    {"mock_scores_op": [3]}
                ]
            },
        ]
    }
    extracted = await LogicEvaluator.extract_conditions(condition)
    # Should flatten out to leaf-level ops
    # => [{'mock_op': [1]}, {'mock_op': [2]}, {'mock_scores_op': [3]}]
    assert len(extracted) == 3
    # confirm each is a dict with single key
    ops = [list(item.keys())[0] for item in extracted]
    assert ops == ["mock_op", "mock_op", "mock_scores_op"]


def test_sort_conditions(evaluator):
    """
    Tests that conditions under and/or are sorted such that the sub-condition
    containing an operation from evaluator.gpu_accelerable_operations is placed last
    or first as the code indicates.
    In logic_evaluator.py, the sort key is: any(k in GPU_ops for k in cond),
    which means non-GPU ops come first (False < True).
    """
    condition = {
        "and": [
            {"mock_op": [1]},      # in GPU ops
            {"some_other_op": [2]},# not in GPU ops
        ]
    }
    sorted_cond = evaluator.sort_conditions(condition)
    # "some_other_op" is not GPU-accelerable => its key is False from the sort
    # "mock_op" => True from the sort => goes last by default
    # => final order: [{'some_other_op': [2]}, {'mock_op': [1]}]
    assert sorted_cond["and"][0] == {"some_other_op": [2]}
    assert sorted_cond["and"][1] == {"mock_op": [1]}


@pytest.mark.asyncio
async def test_rehydrate_single_manifest_basic():
    manifest = {
        "filter": 123,
        "other_field": "unchanged"
    }
    condition_map = {
        123: {
            "operator_name": "mock_op",
            "condition_parameters": [2, 3],
        }
    }
    result = await LogicEvaluator.rehydrate_single_manifest(manifest, condition_map)
    # The filter key is replaced by { "mock_op": [2, 3] }
    assert result["filter"] == {"mock_op": [2, 3]}
    assert result["other_field"] == "unchanged"


@pytest.mark.asyncio
async def test_rehydrate_single_manifest_nested():
    manifest = {
        "filter": {
            "and": [123, {"or": [456, 789]}]
        }
    }
    condition_map = {
        123: {"operator_name": "mock_op", "condition_parameters": [1]},
        456: {"operator_name": "mock_scores_op", "condition_parameters": [9]},
        789: {"operator_name": "mock_op", "condition_parameters": [2, 3]},
    }
    rehydrated = await LogicEvaluator.rehydrate_single_manifest(manifest, condition_map)
    # rehydrated => {
    #   "filter": {
    #     "and": [
    #       {"mock_op": [1]},
    #       {
    #         "or": [
    #            {"mock_scores_op": [9]},
    #            {"mock_op": [2, 3]}
    #         ]
    #       }
    #     ]
    #   }
    # }
    and_part = rehydrated["filter"]["and"]
    assert and_part[0] == {"mock_op": [1]}
    or_part = and_part[1]["or"]
    assert or_part[0] == {"mock_scores_op": [9]}
    assert or_part[1] == {"mock_op": [2, 3]}


@pytest.mark.asyncio
async def test_rehydrate_single_manifest_no_manifest():
    result = await LogicEvaluator.rehydrate_single_manifest(None, {})
    assert result == {}


@pytest.mark.asyncio
async def test_rehydrate_single_manifest_missing_condition():
    manifest = {"filter": 999}
    with pytest.raises(ValueError, match="Condition ID 999 not found in condition_map."):
        await LogicEvaluator.rehydrate_single_manifest(manifest, {})
