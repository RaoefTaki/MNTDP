import ray

@ray.remote
class ValueActor:
    def __init__(self, val):
        self._value = val

    def get(self):
        return self._value

    def set(self, val):
        self._value = val
