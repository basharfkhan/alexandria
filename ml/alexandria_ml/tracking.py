"""Thin experiment-tracking wrapper: MLflow when installed, JSON files otherwise."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from alexandria_ml.config import ML_ROOT

log = logging.getLogger(__name__)


def _clean(name: str) -> str:
    # MLflow metric keys may not contain '@'.
    return re.sub(r"[^\w\-. /]", "_at_", name)


class Tracker:
    def __init__(self, experiment: str = "alexandria", run_name: str | None = None):
        self.params: dict = {}
        self.metrics: dict = {}
        self._mlflow = None
        if os.getenv("ALEXANDRIA_DISABLE_MLFLOW") == "1":
            return
        try:
            import mlflow
        except ImportError:
            log.info("mlflow not installed - logging run to JSON only")
            return
        # MLflow 3 needs a database-backed store; a local SQLite file is the zero-setup option.
        uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ML_ROOT / 'mlflow.db').resolve().as_posix()}")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow
        log.info("MLflow run started (tracking uri: %s)", uri)

    def log_params(self, params: dict) -> None:
        params = {_clean(k): v for k, v in params.items()}
        self.params.update(params)
        if self._mlflow:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], prefix: str = "", step: int | None = None) -> None:
        clean = {_clean(f"{prefix}{k}"): float(v) for k, v in metrics.items()}
        if step is None:
            self.metrics.update(clean)
        if self._mlflow:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifacts(self, path: Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifacts(str(path), artifact_path="model")

    def finish(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run.json").write_text(
            json.dumps({"finished_at": time.time(), "params": self.params, "metrics": self.metrics}, indent=2),
            encoding="utf-8",
        )
        if self._mlflow:
            self._mlflow.end_run()
