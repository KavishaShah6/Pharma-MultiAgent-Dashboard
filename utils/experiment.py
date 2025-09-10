from contextlib import contextmanager
from typing import Dict, Any, Optional

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False


@contextmanager
def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            yield
    else:
        yield


def log_params(params: Dict[str, Any]):
    if _MLFLOW_AVAILABLE:
        mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    if _MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str):
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(path)
        except Exception:
            pass


