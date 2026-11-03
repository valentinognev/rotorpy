"""Recording decoupled from the control path: a bounded queue plus a worker thread."""

import threading

from rotorpy.guidance.recording_worker import RecordingWorker


def test_submitted_items_eventually_reach_the_sink():
    received = []
    done = threading.Event()

    def sink(item):
        received.append(item)
        if item == "last":
            done.set()

    worker = RecordingWorker(sink, maxsize=4)
    worker.submit("a")
    worker.submit("b")
    worker.submit("last")
    assert done.wait(timeout=5.0)
    worker.close(timeout=5.0)
    assert received == ["a", "b", "last"]


def test_submit_does_not_block_while_the_sink_is_slow():
    sink_entered = threading.Event()
    release_sink = threading.Event()
    received = []

    def slow_sink(item):
        sink_entered.set()
        release_sink.wait(timeout=5.0)
        received.append(item)

    worker = RecordingWorker(slow_sink, maxsize=4)
    worker.submit("a")
    assert sink_entered.wait(timeout=5.0)
    # The sink is now blocked inside slow_sink("a"). submit() must still return immediately
    # instead of waiting for the sink, so a slow recorder cannot stall the control path.
    submitted = threading.Event()

    def submit_more():
        worker.submit("b")
        worker.submit("c")
        submitted.set()

    submitter = threading.Thread(target=submit_more)
    submitter.start()
    assert submitted.wait(timeout=5.0)
    submitter.join(timeout=5.0)
    release_sink.set()
    worker.close(timeout=5.0)
    assert received[0] == "a"


def test_a_full_queue_drops_the_oldest_item_and_counts_it():
    release_sink = threading.Event()
    sink_entered = threading.Event()

    def blocking_sink(item):
        if item == "first":
            sink_entered.set()
            release_sink.wait(timeout=5.0)

    worker = RecordingWorker(blocking_sink, maxsize=2)
    worker.submit("first")  # picked up by the worker thread immediately, blocking it
    assert sink_entered.wait(timeout=5.0)
    worker.submit("old")
    worker.submit("newer")
    worker.submit("newest")  # queue (maxsize=2) is full: "old" must be dropped, not "newer"
    assert worker.dropped == 1
    release_sink.set()
    worker.close(timeout=5.0)


def test_close_stops_the_worker_thread():
    worker = RecordingWorker(lambda item: None, maxsize=4)
    worker.close(timeout=5.0)
    assert worker.is_alive() is False


def test_a_failing_sink_does_not_crash_the_worker_or_the_caller():
    processed = threading.Event()

    def failing_sink(item):
        if item == "boom":
            raise ValueError("sink failed")
        processed.set()

    worker = RecordingWorker(failing_sink, maxsize=4)
    worker.submit("boom")
    worker.submit("ok")
    assert processed.wait(timeout=5.0)
    worker.close(timeout=5.0)
    assert worker.failed_count == 1
    assert isinstance(worker.last_error, ValueError)
