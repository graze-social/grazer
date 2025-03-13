import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime
import uuid

# Fully mock `ray.remote` before importing `CPUWorker`
with patch("ray.remote", lambda *args, **kwargs: lambda cls: cls):
    from app.ray.cpu_worker import CPUWorker  # Import CPUWorker after patching


@pytest.fixture(scope="function")
def cpu_worker_instance(mocker):
    """
    Provide a CPUWorker instance with Ray removed.
    Mocks out dependencies on GPU workers, network workers, and cache.
    """
    print("Setting up mock CPUWorker instance...")

    # Mock dependencies
    mock_gpu_embedding_workers = [AsyncMock()]
    mock_gpu_classifier_workers = [AsyncMock()]
    mock_network_workers = [AsyncMock()]
    mock_cache = AsyncMock()

    # Return a direct instance
    instance = CPUWorker(
        gpu_embedding_workers=mock_gpu_embedding_workers,
        gpu_classifier_workers=mock_gpu_classifier_workers,
        network_workers=mock_network_workers,
        cache=mock_cache
    )
    return instance


@pytest.mark.asyncio
async def test_max_concurrency(cpu_worker_instance):
    print("Running test_max_concurrency...")
    result = await cpu_worker_instance.max_concurrency()
    assert result == 5


@pytest.mark.asyncio
async def test_get_active_task_count(cpu_worker_instance):
    print("Running test_get_active_task_count...")
    result = await cpu_worker_instance.get_active_task_count()
    assert result == 0


@pytest.mark.asyncio
async def test_process_manifest_success(cpu_worker_instance, mocker):
    print("Running test_process_manifest_success...")

    algorithm_id = "test-algo"
    manifest = {"config": "mock-config"}
    records = [{"data": "mock-record"}]

    # Mock AlgoManager
    mock_algo_manager = AsyncMock()
    mock_algo_manager.is_gpu_accelerated.return_value = False
    mock_algo_manager.matching_records.return_value = (["matched-record"], None, 0.123)

    mocker.patch("app.algos.manager.AlgoManager.initialize", return_value=mock_algo_manager)

    await cpu_worker_instance.process_manifest(algorithm_id, manifest, records)

    cpu_worker_instance.cache.report_output.remote.assert_called_once()
    response = cpu_worker_instance.cache.report_output.remote.call_args[0][0]

    assert response["algorithm_id"] == algorithm_id
    assert response["compute_environment"] == "cpu"
    assert response["compute_amount"] == 0.123
    assert response["matches"] == ["matched-record"]


@pytest.mark.asyncio
async def test_process_manifest_failure(cpu_worker_instance, mocker):
    print("Running test_process_manifest_failure...")

    algorithm_id = "test-algo"
    manifest = {"config": "mock-config"}
    records = [{"data": "mock-record"}]

    # Simulate exception in AlgoManager
    mocker.patch("app.algos.manager.AlgoManager.initialize", side_effect=Exception("Test error"))

    await cpu_worker_instance.process_manifest(algorithm_id, manifest, records)

    cpu_worker_instance.cache.report_output.remote.assert_called_once()
    response = cpu_worker_instance.cache.report_output.remote.call_args[0][0]

    assert response["algorithm_id"] == algorithm_id
    assert response["compute_environment"] == "none"
    assert response["compute_amount"] == 0
    assert "error" in response


@pytest.mark.asyncio
async def test_process_batch(cpu_worker_instance, mocker):
    print("Running test_process_batch...")

    records = [{"data": "record1"}, {"data": "record2"}]
    manifests = [("algo1", {"config": "config1"}), ("algo2", {"config": "config2"})]

    mock_process_manifest = AsyncMock()
    mocker.patch.object(cpu_worker_instance, "process_manifest", new=mock_process_manifest)

    await cpu_worker_instance.process_batch(records, manifests)

    assert mock_process_manifest.call_count == 2
    mock_process_manifest.assert_any_call("algo1", {"config": "config1"}, records)
    mock_process_manifest.assert_any_call("algo2", {"config": "config2"}, records)
