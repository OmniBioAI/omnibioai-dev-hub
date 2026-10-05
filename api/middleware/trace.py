"""
OmniBioAI api.middleware.trace.

Purpose:
    Defines TraceContext with start, add_step and get methods for api.middleware.trace.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

import time
import uuid


class TraceContext:
    def __init__(self):
        self.store: dict[str, dict] = {}

    def start(self, query: str):
        trace_id = str(uuid.uuid4())

        self.store[trace_id] = {
            "query": query,
            "start_time": time.time(),
            "steps": []
        }

        return trace_id

    def add_step(self, trace_id: str, step: str, data=None):
        if trace_id not in self.store:
            return

        self.store[trace_id]["steps"].append({
            "step": step,
            "data": data,
            "time": time.time()
        })

    def get(self, trace_id: str):
        return self.store.get(trace_id, {})