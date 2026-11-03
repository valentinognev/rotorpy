"""Recording decoupled from the control path: a bounded queue plus a worker thread.

`RecordingWorker.submit` never blocks the caller on a slow sink: it enqueues
and returns immediately. If the bounded queue is full it drops the oldest
queued item (not the newest), counting the drop, so recording latency never
backs up and cannot stall setpoint publishing. A background thread drains the
queue and calls the sink; a failing sink is caught and counted, never raised
into the caller or allowed to stop the worker.
"""

import queue
import threading
from typing import Any, Callable, Optional

# Proposed, not verified: how many queued recording items to hold before dropping the oldest.
PROPOSED_RECORDING_QUEUE_DEPTH = 8

_STOP = object()


class RecordingWorker(object):
    def __init__(
        self,
        sink: Callable[[Any], None],
        maxsize: int = PROPOSED_RECORDING_QUEUE_DEPTH,
        daemon: bool = True,
    ) -> None:
        self._sink = sink
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.failed_count = 0
        self.last_error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=daemon)
        self._thread.start()

    def submit(self, item: Any) -> None:
        """Enqueue one item without blocking. Drops the oldest queued item if full."""
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Another producer raced us and refilled the queue; drop this item too.
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._sink(item)
            except Exception as exc:  # the sink's failure must never stop recording or control
                self.failed_count += 1
                self.last_error = exc

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop the worker once queued items so far have drained."""
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
