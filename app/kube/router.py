from app.kube.processor import KubeProcessor

class RunpodRouter:
    @classmethod
    async def process_request(cls, dispatcher, params):
        print("Params are here!")
        print(type(params))
        print(params.keys())
        if params.get("task") == "process_algos":
            await KubeProcessor.process_algos(dispatcher, params.get("transactions"))
