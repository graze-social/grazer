import ray

@ray.remote
class GrazeBoi:
    def __init__(self, value):
        self.value = value

    def whichboi(self):
        return self.value

def main():
    ray.init(
        ignore_reinit_error=True,
        namespace="default",
    )
    gb1 = GrazeBoi.options(name="gb1", get_if_exists=True, lifetime="detached").remote("bigboi number one")
    assert ray.get(gb1.whichboi.remote()) == "bigboi number one"

    gb2 = GrazeBoi.options(name="gb2", get_if_exists=True, lifetime="detached").remote("bigboi number two")
    assert ray.get(gb2.whichboi.remote()) == "bigboi number two"


if __name__ == "__main__":
    main()
