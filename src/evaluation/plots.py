#src/evaluation/plots.py

import matplotlib.pyplot as plt

def plot_equity_curve(equity_history, trades, save_path: str):
    """
    Plot equity curve with optional trade markers.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(equity_history, label="Equity", color="blue")

    # Mark trades if available
    for t in trades:
        if t is None or "type" not in t:
            continue
        if t["type"] == "buy":
            plt.scatter(t["step"], t["equity"], marker="^", color="green", label="Buy")
        elif t["type"] == "sell":
            plt.scatter(t["step"], t["equity"], marker="v", color="red", label="Sell")

    plt.title("Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
