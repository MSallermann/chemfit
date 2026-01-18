import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FOLDER = Path("./results")

fig, ax = plt.subplots()

for i, p in enumerate(RESULTS_FOLDER.glob("*.json")):
    with p.open("r") as f:
        res = json.load(f)

    wtimes = np.array(res["params"]["wait_times"])
    time_taken = np.array(res["time_taken_list"])
    ax.plot(
        wtimes,
        time_taken,
        label=res["params"]["label"],
        marker="s",
        mec="black",
        ls="None",
        color=f"C{i}",
    )

    wtimes_cont = np.linspace(np.min(wtimes), np.max(wtimes), 500)

    ax.plot(
        wtimes_cont,
        wtimes_cont * res["time_slope"] + res["time_offset"],
        marker="None",
        ls="-",
        color=f"C{i}",
    )

    # def cost_func(x: float, a: float, b: float):
    #     return a * x + b

    # popt, pcov = curve_fit(
    #     cost_func,
    #     wtimes,
    #     time_taken,
    #     sigma=time_taken / np.log(time_taken), # this creates an even weight in log-space
    #     absolute_sigma=True,
    # )

    # ax.plot(
    #     wtimes,
    #     cost_func(wtimes, *popt),
    #     marker="None",
    #     ls="-",
    #     color=f"C{i}",
    # )

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_ylabel("Total time [s]")
ax.set_xlabel("Time per task [s]")
ax.grid(visible=True, which="both", color="lightgrey", ls="-")
fig.savefig("time_taken.png", dpi=300)
