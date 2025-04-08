import pytest
import random
import asyncio
from unittest.mock import AsyncMock, patch
from app.ray.dispatcher import Dispatcher

# Fully mock `ray.remote` before importing `CPUWorker`
with patch("ray.remote", lambda *args, **kwargs: lambda cls: cls):
    from app.ray.cpu_worker import CPUWorker  # Import CPUWorker after patching

@pytest.fixture(scope="function")
def dispatcher_instance(mocker):
    """
    Provide a Dispatcher instance with all actors mocked.
    """
    print("Setting up mock Dispatcher instance...")

    # Mock actor discovery
    mock_actor = AsyncMock()
    mocker.patch("app.ray.utils.discover_named_actor", return_value=mock_actor)
    mocker.patch("app.ray.utils.discover_named_actors", return_value=[mock_actor])

    # Return an instance of Dispatcher (now using mocked actors)
    mock_cache = AsyncMock()
    mock_bluesky_semaphore = AsyncMock()
    mock_graze_semaphore = AsyncMock()
    mock_network_workers = [AsyncMock()]
    mock_gpu_embedding_workers = [AsyncMock()]
    mock_gpu_classifier_workers = [AsyncMock()]
    mock_cpu_workers = [AsyncMock()]
    
    instance = Dispatcher(
        cache=mock_cache,
        bluesky_semaphore=mock_bluesky_semaphore,
        graze_semaphore=mock_graze_semaphore,
        network_workers=mock_network_workers,
        gpu_embedding_workers=mock_gpu_embedding_workers,
        gpu_classifier_workers=mock_gpu_classifier_workers,
        cpu_workers=mock_cpu_workers,
    )
    return instance


@pytest.mark.asyncio
async def test_generate_timing_report(dispatcher_instance):
    """
    Test that generate_timing_report correctly collects and sorts timing data.
    """
    print("Running test_generate_timing_report...")

    # Mock responses for `get_summary.remote()`
    sample_timings = {
        "method_1": 2.5,
        "method_2": 0.8,
        "method_3": 4.1,
    }
    dispatcher_instance.bluesky_semaphore.get_summary.remote.return_value = sample_timings
    dispatcher_instance.graze_semaphore.get_summary.remote.return_value = sample_timings

    # Mock network workers
    for worker in dispatcher_instance.network_workers:
        worker.get_summary.remote.return_value = sample_timings

    # Mock GPU workers
    for worker in dispatcher_instance.gpu_embedding_workers:
        worker.get_summary.remote.return_value = sample_timings
    for worker in dispatcher_instance.gpu_classifier_workers:
        worker.get_summary.remote.return_value = sample_timings

    # Mock CPU workers
    for worker in dispatcher_instance.cpu_workers:
        worker.get_summary.remote.return_value = sample_timings

    # Run the method
    report = await dispatcher_instance.generate_timing_report()

    # Ensure sorting is correct (highest timing first)
    expected_order = ['bluesky_semaphore.method_3', 'graze_semaphore.method_3', 'network_worker_0.method_3', 'gpu_embedding_worker_0.method_3', 'gpu_classifier_worker_0.method_3', 'cpu_worker_0.method_3', 'bluesky_semaphore.method_1', 'graze_semaphore.method_1', 'network_worker_0.method_1', 'gpu_embedding_worker_0.method_1', 'gpu_classifier_worker_0.method_1', 'cpu_worker_0.method_1', 'bluesky_semaphore.method_2', 'graze_semaphore.method_2', 'network_worker_0.method_2', 'gpu_embedding_worker_0.method_2', 'gpu_classifier_worker_0.method_2', 'cpu_worker_0.method_2']
    assert list(report.keys()) == expected_order


@pytest.mark.asyncio
async def test_distribute_tasks(dispatcher_instance, mocker):
    """
    Test that distribute_tasks assigns tasks to available CPU workers.
    """
    print("Running test_distribute_tasks...")

    # Mock CPU workers with different concurrency limits
    cpu_workers = [AsyncMock() for _ in range(3)]
    dispatcher_instance.cpu_workers = cpu_workers

    # Configure CPU worker concurrency
    for i, worker in enumerate(cpu_workers):
        worker.max_concurrency.remote.return_value = 5
        worker.get_active_task_count.remote.return_value = i  # Different active loads

    # Define test input
    records = [{"data": "mock-record"}]
    manifest_data = [{"config": "mock-config"}]

    # Run the method
    await dispatcher_instance.distribute_tasks(records, manifest_data)

    # Ensure at least one worker was assigned the task
    assigned_worker = any(
        worker.process_batch.remote.called for worker in cpu_workers
    )
    assert assigned_worker


@pytest.mark.asyncio
async def test_distribute_tasks_with_busy_workers(dispatcher_instance, mocker):
    """
    Ensure distribute_tasks waits for availability if all workers are busy.
    """
    print("Running test_distribute_tasks_with_busy_workers...")

    # Mock CPU workers (all at full capacity)
    cpu_workers = [AsyncMock() for _ in range(3)]
    dispatcher_instance.cpu_workers = cpu_workers

    # Set all workers to full capacity
    for worker in cpu_workers:
        worker.max_concurrency.remote.return_value = 5
        worker.get_active_task_count.remote.return_value = 5  # Fully occupied

    # Define test input
    records = [{"data": "mock-record"}]
    manifest_data = [{"config": "mock-config"}]

    # Patch asyncio.sleep to avoid real delays
    with patch("asyncio.sleep", new=AsyncMock()):
        # Run the method (should loop until a worker is available)
        distribute_task = asyncio.create_task(
            dispatcher_instance.distribute_tasks(records, manifest_data)
        )
        await asyncio.sleep(0.2)  # Allow some event loop cycles

        # Free up a worker
        cpu_workers[0].get_active_task_count.remote.return_value = 3  # Now available

        # Allow time for method to recognize free worker
        await distribute_task

    # Ensure the task was finally assigned
    cpu_workers[0].process_batch.remote.assert_called_once_with(records, manifest_data)


