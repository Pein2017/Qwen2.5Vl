import io
import json

from src_post.logging.logging_utils import rank0_log


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        pass


class _DummyTB:
    def add_scalar(self, *args, **kwargs):
        pass


class _DummyWriter:
    def __init__(self):
        self.buf = io.StringIO()
    def write(self, s: str):
        self.buf.write(s)


def test_rank0_log_writes_accuracy_and_fn_rate():
    logger = _DummyLogger()
    tb = _DummyTB()
    writer = _DummyWriter()
    scalars = {
        "loss": 1.0,
        "reward_best_mean": 0.1,
        "reward_best_std": 0.05,
        "acc_best": 0.8,
        "acc_any": 0.9,
        "accuracy": 0.8,
        "fn_rate": 0.2,
        "resp_len_mean": 10.0,
        "grad_norm_mean": 0.3,
        "eta_hours": 0.1,
    }
    lrs = {"aligner": 1e-5}
    rank0_log(logger, tb, writer, step=1, prefix="[test] ", scalars=scalars, lrs=lrs)
    s = writer.buf.getvalue().strip().splitlines()[-1]
    rec = json.loads(s)
    assert isinstance(rec.get("accuracy"), (int, float))
    assert isinstance(rec.get("fn_rate"), (int, float))
