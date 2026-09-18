"""
Practical 1: load UART-logged displacement data and plot y(t).

Expected data format in ../data/data.txt (two columns, space-separated):
    t[s]  y[m]

Example:
    0.0000 0.0000
    0.1000 0.0109
    ...
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.txt"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Save PuTTY session output to data/data.txt first."
        )

    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected two columns (t y) in {path}, got shape {data.shape}"
        )

    t = data[:, 0]
    y = data[:, 1]
    return t, y


def plot_displacement(t: np.ndarray, y: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y, "o-", color="tab:blue", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title("Залежність переміщення від часу")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()


def numerical_derivative(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Чисельна похідна f(t): центральна різниця у внутрішніх точках
    (i = 1..n-2, використовує сусідів з обох боків, похибка O(dt^2)),
    одностороння різниця на краях (i = 0 і i = n-1, де сусід є лише
    з одного боку, похибка O(dt))."""
    d = np.empty_like(f, dtype=float)
    d[0] = (f[1] - f[0]) / (t[1] - t[0])
    d[-1] = (f[-1] - f[-2]) / (t[-1] - t[-2])
    d[1:-1] = (f[2:] - f[:-2]) / (t[2:] - t[:-2])
    return d


def compute_velocity(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """швидкість v(t) як перша похідна переміщення y(t)."""
    return numerical_derivative(t, y)


def compute_acceleration(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """прискорення a(t) як друга похідна переміщення (перша
    похідна v(t))."""
    return numerical_derivative(t, v)

def plot_velocity(t: np.ndarray, v: np.ndarray, save_path: Path) -> None:
    """Plot velocity v(t) and save to file."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, v, "o-", color="tab:orange", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Залежність швидкості від часу")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()
def plot_acceleration(t: np.ndarray, a: np.ndarray, save_path: Path) -> None:
    """Plot acceleration a(t) and save to file."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, a, "o-", color="tab:green", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Залежність прискорення від часу")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()



def main() -> None:
    t, y = load_data(DATA_PATH)
    print(f"Loaded {len(t)} samples from {DATA_PATH}")
    print(np.column_stack((t, y)))

    plot_displacement(t, y, OUTPUT_DIR / "displacement_plot.png")
    plot_velocity(t, compute_velocity(t, y), OUTPUT_DIR / "velocity_plot.png")
    plot_acceleration(t, compute_acceleration(t, compute_velocity(t, y)), OUTPUT_DIR / "acceleration_plot.png")




if __name__ == "__main__":
    main()
