"""CUDA loss logging copies to pinned host memory without per-interval synchronization."""

from collections import deque

import pytest
import torch

from swapface.train import Trainer


class CaptureWriter:
    def __init__(self):
        self.records: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, key: str, value: float, step: int) -> None:
        self.records.append((key, value, step))

    def close(self) -> None:
        self.closed = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_loss_logging_async_copy_preserves_step_order_and_flushes_on_close():
    trainer = Trainer.__new__(Trainer)
    trainer._pending_logs = deque()
    trainer._log_buffer = {}
    trainer._completed_step = 9
    trainer.log_interval = 10
    writer = CaptureWriter()
    trainer.log_writer = writer

    trainer.log("g_loss", torch.tensor([2.0, 4.0], device="cuda"))
    trainer.log("d_loss", torch.tensor(1.5, device="cuda"))
    trainer._completed_step = 10
    trainer.flush_logs()
    assert writer.records == []  # enqueue only; don't wait for the GPU
    assert len(trainer._pending_logs) == 1
    ready, step, keys, host_values = trainer._pending_logs[0]
    assert step == 10 and keys == ("g_loss", "d_loss")
    assert host_values.is_pinned() and host_values.device.type == "cpu"

    # The next batch of metrics drains the previous completed D2H copy.
    ready.synchronize()  # synchronize only in the test, not in the training loop
    trainer._completed_step = 15
    trainer.log("r1_loss", torch.tensor(6.0, device="cuda"), force=True)
    trainer._completed_step = 16
    trainer.flush_logs()
    assert writer.records == [("Loss/g_loss", 3.0, 10), ("Loss/d_loss", 1.5, 10)]
    assert len(trainer._pending_logs) == 1

    trainer.close_logs()
    assert writer.records == [
        ("Loss/g_loss", 3.0, 10),
        ("Loss/d_loss", 1.5, 10),
        ("Loss/r1_loss", 6.0, 16),
    ]
    assert len(trainer._pending_logs) == 0
    assert writer.closed


def test_unfinished_gpu_log_copy_is_not_read_early():
    trainer = Trainer.__new__(Trainer)
    writer = CaptureWriter()
    trainer.log_writer = writer

    class DelayedEvent:
        def __init__(self):
            self.done = False
            self.waits = 0

        def query(self):
            return self.done

        def synchronize(self):
            self.waits += 1
            self.done = True

    event = DelayedEvent()
    trainer._pending_logs = deque([(event, 3, ("d_loss",), torch.tensor([1.25]))])
    trainer._write_ready_logs()
    assert writer.records == [] and event.waits == 0
    assert len(trainer._pending_logs) == 1

    trainer.close_logs()
    assert event.waits == 1
    assert writer.records == [("Loss/d_loss", 1.25, 3)]
    assert writer.closed
