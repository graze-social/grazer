import pytest
import ray

@pytest.fixture(scope="session")
def ray_session():
    ray.init(
        num_cpus=2,
        num_gpus=0,  # Ensure Ray does not allocate GPU resources
        object_store_memory=512 * 1024 * 1024,  # Prevent memory misallocation
        include_dashboard=False,
        local_mode=False,  # Ensure normal execution mode
        logging_level="DEBUG"
    )
    yield
    ray.shutdown()

@pytest.fixture(scope="function")
def cache_actor(ray_session):
    from app.ray.cache import Cache
    actor = Cache.remote(batch_size=2)
    yield actor
    ray.kill(actor)
