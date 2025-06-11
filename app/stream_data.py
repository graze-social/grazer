import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StreamTransaction:
    receipt_handle: str
    body: Dict[str, Any]

    @property
    def without_receipt(self) -> Dict[str, Any]:
        return self.body


@dataclass
class StreamData:
    data: Dict[str, Any]
    input: Optional[Dict[str, Any]] = None
    # TBD: follow the previous logic for doing backfills, etc
    # See app/runpod/router.py
    task: str = "process_algos"

    def __post_init__(self) -> None:
        self.input = {"task": self.task, "transactions": [self.data]}

    def __repr__(self) -> str:
        return json.dumps(self.wrap())

    def wrap(self) -> Dict[str, Any]:
        return {"input": {"task": self.task, "transactions": [self.data]}}

    def transactions(self) -> List[Dict[str, Any]]:
        """Return receipt w/o receipt handle for processing"""
        return [self.data]

    # def receipt_handles(self) -> List[str]:
    #     return [datum.receipt_handle for datum in self.data]
