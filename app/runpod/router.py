from app.runpod.processor import RunpodProcessor
from app.runpod.auditor import RunpodAuditor
from app.runpod.backfiller import RunpodBackfiller
from app.runpod.backtester import RunpodBacktester


class RunpodRouter:
    @classmethod
    async def process_request(cls, dispatcher, params):
        print("Params are here!")
        print(type(params))
        print(params.keys())
        if params.get("task") == "process_algos":
            await RunpodProcessor.process_algos(dispatcher, params.get("transactions"))
        elif params.get("task") == "run_backtest":
            await RunpodBacktester.live_query(
                dispatcher, params.get("task_id"), params.get("manifest")
            )
        elif params.get("task") == "run_backfill":
            await RunpodBackfiller.run(
                dispatcher, params.get("algorithm_id"), params.get("manifest")
            )
        elif params.get("task") == "debug_post":
            print("About to debug with the following params:")
            print(params)
            await RunpodAuditor.audit_records(
                dispatcher,
                params.get("task_id"),
                params.get("manifest"),
                params.get("records"),
            )
