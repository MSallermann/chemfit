import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_FOLDER = Path("./results")

fig, ax = plt.subplots()

for i, p in enumerate(RESULTS_FOLDER.glob("*.json")):
    with p.open("r") as f:
        res = json.load(f)

    wtimes = res["params"]["wait_times"]
    ax.plot(
        wtimes,
        res["time_taken_list"],
        label=res["params"]["label"],
        marker=".",
        ls="-",
        color=f"C{i}",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_ylabel("Total time [s]")
ax.set_xlabel("Time per task [s]")
ax.grid(visible=True, which="both", color="lightgrey", ls="-")
fig.savefig("time_taken.png", dpi=300)
