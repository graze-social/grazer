from app.runpod.base import RunpodBase


class RunpodAuditor(RunpodBase):
    @classmethod
    async def run_audit(cls, records, manager, task_id):
        try:
            return await manager.audit_records(records)
        except Exception:
            await cls.publish_status(
                task_id,
                {
                    "error": "Algorithm couldn't analyze these records - please check your logic and the posts you provided."
                },
            )

    @classmethod
    async def audit_records(cls, dispatcher, task_id, manifest, records):
        await cls.publish_status(
            task_id, {"status": "Job has been received by a live feed analyzer..."}
        )
        await cls.publish_status(task_id, {"status": "Loading feed analyzer..."})
        manager = await cls.initialize_algo(dispatcher, manifest, task_id)
        if not manager:
            return
        await cls.publish_status(
            task_id, {"status": "Starting audit on provided posts..."}
        )
        audit_log = await cls.run_audit(records, manager, task_id)
        if audit_log:
            await cls.publish_status(task_id, {"audit_log": audit_log})
        else:
            await cls.publish_status(
                task_id,
                {
                    "error": "No Audit Log was Returned! Likely an error internal to the audit procedure."
                },
            )
        await cls.publish_status(task_id, {"finished": True})
