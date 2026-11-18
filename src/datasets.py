"""Where the trip folders for prediction live (used by the API and run_eval.py).

Each dataset is a folder laid out like the provided data: <folder>/<trip_id>/<trip_id>.track etc.
The evaluator folder can be moved with the PATHSNAP_EVAL_DIR environment variable.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ORDER = ["evaluator_test", "student_test"]   # first one that exists is the default


def dataset_folders():
    return {
        "student_test": os.path.join(ROOT, "data", "raw", "Student folder", "student_test"),
        "evaluator_test": os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "raw", "evaluator_test")),
    }


def available_datasets():
    return {name: path for name, path in dataset_folders().items() if os.path.isdir(path)}


def trip_ids(folder):
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
