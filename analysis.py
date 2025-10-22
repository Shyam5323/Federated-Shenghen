import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CENTRALIZED_AUC = 0.8625  

MAX_ROUNDS = 15

iid_data = [
    {"round": 1, "auc": 0.8232},
    {"round": 2, "auc": 0.8582},
    {"round": 3, "auc": 0.8686},
    {"round": 4, "auc": 0.8714},
    {"round": 5, "auc": 0.8722},
    {"round": 6, "auc": 0.8718},
    {"round": 7, "auc": 0.8766},
    {"round": 8, "auc": 0.8750},
    {"round": 9, "auc": 0.8765},
    {"round": 10, "auc": 0.8748},
    {"round": 11, "auc": 0.8761},
    {"round": 12, "auc": 0.8753},
    {"round": 13, "auc": 0.8745},
    {"round": 14, "auc": 0.8758},
    {"round": 15, "auc": 0.8749}
]

label_skew_data = [
    {"round": 1, "auc": 0.8054302864917213}, {"round": 2, "auc": 0.8375548419727818},
    {"round": 3, "auc": 0.8341886524655854}, {"round": 4, "auc": 0.8367359861361061},
    {"round": 5, "auc": 0.8429525859533974}, {"round": 6, "auc": 0.8437751665353205},
    {"round": 7, "auc": 0.8443080011448688}, {"round": 8, "auc": 0.8470298097243296},
    {"round": 9, "auc": 0.8422233200418738}, {"round": 10, "auc": 0.8402292090601488},
    {"round": 11, "auc": 0.8478257289718525}, {"round": 12, "auc": 0.8413280089080224},
    {"round": 13, "auc": 0.8381327656037421}, {"round": 14, "auc": 0.8452533022807204},
    {"round": 15, "auc": 0.8366409071126949}
]

pathological_data = [
    {"round": 1, "auc": 0.7857220712720201}, {"round": 2, "auc": 0.8071860137776367},
    {"round": 3, "auc": 0.8206205817659997}, {"round": 4, "auc": 0.8296238791457394},
    {"round": 5, "auc": 0.8313313807826669}, {"round": 6, "auc": 0.8336658158564364},
    {"round": 7, "auc": 0.834640914954264},  {"round": 8, "auc": 0.834140230777374},
    {"round": 9, "auc": 0.8366377704851184}, {"round": 10, "auc": 0.8266046790641871},
    {"round": 11, "auc": 0.832379406471647}, {"round": 12, "auc": 0.8283488400359145},
    {"round": 13, "auc": 0.8346205268750171}, {"round": 14, "auc": 0.8332911849002749},
    {"round": 15, "auc": 0.8291526008523786}
]

HISTORY_DATA = {
    "IID": iid_data,
    "Label Skew": label_skew_data,
    "Pathological Non-IID": pathological_data,
}


def process_data(data, max_rounds):
    """Extracts rounds and AUCs from the data structure and truncates."""
    try:
        rounds = [item['round'] for item in data]
        aucs = [item['auc'] for item in data]
        
        valid_indices = [i for i, r in enumerate(rounds) if r <= max_rounds]
        return np.array(rounds)[valid_indices], np.array(aucs)[valid_indices]

    except (KeyError, IndexError) as e:
        print(f"Warning: Could not parse data. Check format. Error: {e}. Skipping.")
        return None, None

print("--- Generating Performance Summary Table ---")

results = {
    "Strategy": ["Centralized Benchmark", "Federated IID", "Federated Label Skew", "Federated Pathological Non-IID"],
    "Best Validation AUC": [CENTRALIZED_AUC, 0, 0, 0]
}

federated_data_processed = {}

for name, data in HISTORY_DATA.items():
    rounds, aucs = process_data(data, MAX_ROUNDS)
    if rounds is not None:
        federated_data_processed[name] = (rounds, aucs)
        if len(aucs) > 0:
            best_auc = max(aucs)
            if name == "IID":
                results["Best Validation AUC"][1] = best_auc
            elif name == "Label Skew":
                results["Best Validation AUC"][2] = best_auc
            elif name == "Pathological Non-IID":
                results["Best Validation AUC"][3] = best_auc

df = pd.DataFrame(results)
df["Best Validation AUC"] = df["Best Validation AUC"].round(4)
print(df.to_string(index=False))
print("-" * 40)


print("\n--- Generating Convergence Plot ---")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

for name, (rounds, aucs) in federated_data_processed.items():
    if rounds is not None and len(rounds) > 0:
        ax.plot(rounds, aucs, marker='o', linestyle='-', label=name)

ax.axhline(y=CENTRALIZED_AUC, color='r', linestyle='--', label=f'Centralized Benchmark (AUC={CENTRALIZED_AUC:.4f})')

ax.set_title('Federated Learning Performance on Shenzhen Dataset', fontsize=16)
ax.set_xlabel('Federated Round', fontsize=12)
ax.set_ylabel('Validation AUC', fontsize=12)
ax.legend(fontsize=11)
ax.set_xticks(range(1, MAX_ROUNDS + 1))
ax.set_ylim(bottom=min(df["Best Validation AUC"].min() * 0.95, 0.75))
ax.grid(True)

plot_filename = 'convergence_plot.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved to {plot_filename}")

plt.show()