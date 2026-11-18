"""Small helpers to make terminal output easy to read during the demo."""
import time
import pandas as pd

WIDTH = 78


def banner(title):
    print("\n" + "=" * WIDTH)
    print(title.center(WIDTH))
    print("=" * WIDTH)


def step(number, title):
    print(f"\n--- Step {number}: {title} " + "-" * max(3, WIDTH - 12 - len(str(number)) - len(title)))


def note(text):
    print(f"  > {text}")


def table(df, floatfmt="{:.4f}", index=False):
    """Print a DataFrame with tidy number formatting, indented.
    floatfmt: one format for all float columns, or a dict {column: format}."""
    shown = df.copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            fmt = floatfmt.get(c, "{:.4f}") if isinstance(floatfmt, dict) else floatfmt
            shown[c] = shown[c].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    text = shown.to_string(index=index)
    print("\n".join("  " + line for line in text.splitlines()))


class Timer:
    """with Timer("label"): ...  -> prints how long the block took."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        note(f"{self.label}: {self.seconds:.1f} s")
