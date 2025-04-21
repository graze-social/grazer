from app.kube.processor import KubeProcessor

class KubeRouter:
    @classmethod
    async def process_request(cls, dispatcher, params, noop: bool):
        print("Params are here!")
        print(type(params))
        print(params.keys())
        if noop:
            print("noop")
        else:
            if params.get("task") == "process_algos":
                await KubeProcessor.process_algos(dispatcher, params.get("transactions"))
