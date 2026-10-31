"""Diagnostics: a wall-clock timeline of the measured run, printed to stderr (train.log).

`install()` stamps the process age at apply(), around Soup's setup / train / save phases and
at exit, so the time outside the training steps can be attributed. Nothing here changes what
runs.
"""

import atexit
import os
import sys
import time


def process_age():
    """Seconds since this process started (Linux), else since this module was imported."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return time.perf_counter() - _T0


_T0 = time.perf_counter()


def mark(what):
    print(f"[solution-timeline] {process_age():7.2f} s  {what}", file=sys.stderr, flush=True)


def _wrap(owner, name, label):
    fn = owner.__dict__.get(name)
    if not callable(fn) or getattr(fn, "_soup_timeline", False):
        return

    def wrapper(*args, **kwargs):
        mark(f"{label} begin")
        try:
            return fn(*args, **kwargs)
        finally:
            mark(f"{label} end")

    wrapper._soup_timeline = True
    setattr(owner, name, wrapper)


_INSTALLED = []
_SAMPLER = {"stop": None}


def _sample_main_thread(main_id, stop, interval=0.05):
    """Inclusive sample counts of (module, function) on the main thread's stack."""
    import collections

    counts = collections.Counter()
    n = 0
    while not stop.wait(interval):
        frame = sys._current_frames().get(main_id)
        seen = set()
        while frame is not None:
            path = frame.f_code.co_filename.replace("\\", "/")
            for marker in ("site-packages/", "dist-packages/", "/lib/python3"):
                if marker in path:
                    path = path.split(marker, 1)[1]
                    break
            key = f"{path}:{frame.f_code.co_name}"
            if key not in seen:
                seen.add(key)
                counts[key] += 1
            frame = frame.f_back
        n += 1
    top = [f"{c * interval:5.1f}s {k}" for k, c in counts.most_common(45)]
    print(f"[solution-timeline] main thread from apply() to the first step, {n} samples:\n"
          + "\n".join(f"[solution-sample] {line}" for line in top), file=sys.stderr, flush=True)


def stop_sampler():
    stop = _SAMPLER["stop"]
    if stop is not None:
        _SAMPLER["stop"] = None
        stop.set()


def install():
    mark("apply()")
    if _INSTALLED:
        return
    _INSTALLED.append(True)
    atexit.register(mark, "interpreter exit")
    import threading

    stop = threading.Event()
    _SAMPLER["stop"] = stop
    threading.Thread(target=_sample_main_thread, args=(threading.main_thread().ident, stop),
                     name="solution-timeline-sampler", daemon=True).start()
    try:
        from soup_cli.trainer import sft

        for name in ("setup", "_setup_transformers", "_build_trainer", "train"):
            _wrap(sft.SFTTrainerWrapper, name, f"SFTTrainerWrapper.{name}")
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        _wrap(transformers.Trainer, "train", "Trainer.train")
        _wrap(transformers.Trainer, "save_model", "Trainer.save_model")
    except Exception:  # noqa: BLE001
        pass
