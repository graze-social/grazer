import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StreamData:
    data: Dict[str, Any]
    input: Optional[Dict[str, Any]] = None
    task: str = "process_algos"

    def __post_init__(self) -> None:
        self.input = {"task": self.task, "transactions": [self.data]}

    def __repr__(self) -> str:
        return json.dumps(self.wrap())

    def wrap(self) -> Dict[str, Any]:
        return {"input": {"task": self.task, "transactions": [self.data]}}

    def transactions(self) -> List[Dict[str, Any]]:
        return [self.data]
