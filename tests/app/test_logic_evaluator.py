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


def is_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b

@pytest.mark.parametrize("value,threshold,expected_value,expected_threshold", [
    # Numeric threshold, flat value
    ([1, 2, None], 5, np.array([1, 2, 0]), 5),
    ([None, '', 3], 0, np.array([0, 0, 3]), 0),

    # Numeric-like string threshold
    ([None, ''], '0.7', np.array([0, 0]), 0.7),
    ([1.2, None], '3.14', np.array([1.2, 0]), 3.14),

    # String threshold (not numeric-like)
    ([None, '', 'x'], 'foo', np.array(['', '', 'x']), 'foo'),

    # Nested arrays, numeric threshold
    ([[None, 1], [2, '']], 0, np.array([[0, 1], [2, 0]]), 0),

    # Nested arrays, string threshold (non-numeric)
    ([['a', None], ['', 'b']], 'bar', np.array([['a', ''], ['', 'b']]), 'bar'),

    # Threshold is None
    ([None, 1], None, np.array(['', 1]), ''),

    # Threshold is empty string, non-numeric
    ([None, ''], '', np.array(['', '']), ''),

    # Threshold is empty string but castable
    ([None, 1.5], '0.0', np.array([0, 1.5]), 0.0),

    # Threshold is a float, input includes string numbers
    (['1.1', ''], 2.2, np.array(['1.1', 0]), 2.2),  # NOTE: values are not coerced to float

    # Threshold is string int, value is numeric
    ([1, 2, 3], '4', np.array([1, 2, 3]), 4.0),

    # Threshold is numeric but value is stringy
    (['', ''], 1.0, np.array([0, 0]), 1.0),
])
def test_normalize_comparison_inputs(value, threshold, expected_value, expected_threshold):
    result_value, result_threshold = LogicEvaluator._normalize_comparison_inputs(value, threshold)
    assert is_equal(result_value, expected_value)
    assert result_threshold == expected_threshold

@pytest.mark.asyncio
async def test_compare():
    """Test the static compare method with all supported operators and edge cases."""
    # operator ==
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "==", 5), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "==", 4), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array([None]), "==", ''), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array(['']), "==", None), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array(['foo']), "==", None), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array([None, '']), "==", ''), np.array([True, True]))

    # operator !=
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "!=", 4), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "!=", 5), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array([None]), "!=", ''), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array(['x']), "!=", None), np.array([True]))

    # operator >=
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), ">=", 5), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([4]), ">=", None), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array([None]), ">=", 4), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array(['']), ">=", ''), np.array([True]))  # not numeric

    # operator <=
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "<=", 5), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "<=", None), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array([None]), "<=", 5), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([0.2]), "<=", "0.7"), np.array([True]))

    # operator >
    assert np.array_equal(LogicEvaluator.compare(np.array([6]), ">", 5), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), ">", 5), np.array([False]))
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), ">", None), np.array([False]))

    # operator <
    assert np.array_equal(LogicEvaluator.compare(np.array([4]), "<", 5), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array([5]), "<", None), np.array([False]))

    # operator in
    assert all(LogicEvaluator.compare(np.array([1, 2]), "in", [1, 2, 3]))
    assert all(LogicEvaluator.compare(np.array([None, '']), "in", ['', 'x']))
    assert not any(LogicEvaluator.compare(np.array(['a', 'b']), "in", ['x', 'y']))

    # operator not_in
    assert not any(LogicEvaluator.compare(np.array([1, 2]), "not_in", [1, 2, 3]))
    assert all(LogicEvaluator.compare(np.array(['a', 'b']), "not_in", ['x', 'y']))
    assert not any(LogicEvaluator.compare(np.array([None, '']), "not_in", ['', 'a']))

    # mixed numeric with None
    assert all(LogicEvaluator.compare(np.array([None, '', 3]), "in", [3, '', 'x']))
    assert np.array_equal(LogicEvaluator.compare(np.array([None]), "==", None), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array(['']), "==", ''), np.array([True]))
    assert np.array_equal(LogicEvaluator.compare(np.array(['abc']), "!=", ''), np.array([True]))
    assert LogicEvaluator.compare(np.array([1, 2, None]), "!=", '')[2] == False

    # nested list (list of lists) - string values
    nested_str = np.array([['a', None], ['b', '']], dtype=object)
    result_eq_str = LogicEvaluator.compare(nested_str, '==', '')
    assert result_eq_str.shape == (2,)
    assert result_eq_str[0] is np.True_  # None becomes ''
    assert result_eq_str[1] is np.True_  # '' == ''

    result_neq_str = LogicEvaluator.compare(nested_str, '!=', '')
    assert result_neq_str[0] is np.False_  # 'a' != ''
    assert result_neq_str[1] is np.False_

    # nested list (list of lists) - int values
    nested_int = np.array([[1, None], [0, '']], dtype=object)
    result_eq_int = LogicEvaluator.compare(nested_int, '==', '')
    assert result_eq_int.shape == (2,)
    assert result_eq_int[0] is np.True_   # None becomes ''
    assert result_eq_int[1] is np.True_   # '' == ''
    assert result_eq_int[0] is np.True_  # 1 != ''
    assert result_eq_int[1] is np.True_  # 0 != ''

    result_neq_int = LogicEvaluator.compare(nested_int, '!=', '')
    assert result_neq_int[0] is np.False_
    assert result_neq_int[1] is np.False_
    assert result_neq_int[0] is np.False_
    assert result_neq_int[1] is np.False_

    # unknown comparator
    with pytest.raises(ValueError, match="Unknown comparator 'abcd'"):
        LogicEvaluator.compare(np.array([5]), "abcd", 5)

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


def test_rehydrate_single_manifest_basic():
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
    result = LogicEvaluator.rehydrate_single_manifest(manifest, condition_map)
    # The filter key is replaced by { "mock_op": [2, 3] }
    assert result["filter"] == {"mock_op": [2, 3]}
    assert result["other_field"] == "unchanged"


def test_rehydrate_single_manifest_nested():
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
    rehydrated = LogicEvaluator.rehydrate_single_manifest(manifest, condition_map)
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


def test_rehydrate_single_manifest_no_manifest():
    result = LogicEvaluator.rehydrate_single_manifest(None, {})
    assert result == {}


def test_rehydrate_single_manifest_missing_condition():
    manifest = {"filter": 999}
    with pytest.raises(ValueError, match="Condition ID 999 not found in condition_map."):
        LogicEvaluator.rehydrate_single_manifest(manifest, {})
