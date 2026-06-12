import numpy as np
import pandas as pd
from pathlib import Path

def print_stats(durations: list[float]) -> float:
    return np.mean(durations)

def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["timestamp", "duration"]
    )
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    df = df.dropna()

    df["relative_time"] = df["timestamp"] - df["timestamp"].iloc[0]
    df = df[df["relative_time"] <= 60]
    return df

def calc_fps(path: Path) -> float:
    df = load_run(path=path)

    if len(df) == 0:
        return 0.0
    
    elapsed = df["relative_time"].iloc[-1]
    return len(df) / elapsed

def evaluate(root: Path):
    results = []
    for location in root.iterdir():
        if not location.is_dir():
            continue

        for mode in location.iterdir():
            if not mode.is_dir():
                continue

            for model in mode.iterdir():
                if not model.is_dir():
                    continue

                fps_values = []
                for run in model.iterdir():
                    if not run.is_file():
                        continue

                    fps = calc_fps(run)
                    fps_values.append(fps)

                if fps_values:
                    results.append({
                        "location": location.name,
                        "mode": mode.name,
                        "model": model.name,
                        "fps_mean": np.mean(fps_values),
                        "fps_std": np.std(fps_values),
                        "runs": len(fps_values)
                    })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = evaluate(Path("."))
    print(df.to_string(index=False))
    df.to_csv("fps_results.csv", index=False)