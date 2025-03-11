import pytest
import ray

@pytest.fixture(scope="session")
def ray_local():
    ray.init(local_mode=True, num_cpus=2)
    yield
    ray.shutdown()