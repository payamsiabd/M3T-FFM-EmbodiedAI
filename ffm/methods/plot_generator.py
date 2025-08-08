import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from itertools import combinations
import pandas as pd
from scipy.linalg import bandwidth

SEED = 49
global_random_gn = np.random.default_rng(SEED)


def draw_network(servers, users, area_size):
    plt.figure(figsize=(6, 6))
    plt.scatter(servers[:, 0], servers[:, 1], marker='s', label='Edge Servers', s=50)
    plt.scatter(users[:, 0], users[:, 1], marker='o', alpha=0.6, label='Users', s=30)
    # dynamically compute the plotting bounds so nothing gets clipped:
    all_x = np.concatenate([servers[:, 0], users[:, 0]])
    all_y = np.concatenate([servers[:, 1], users[:, 1]])
    margin = 0.05 * max(area_size, all_x.ptp(), all_y.ptp())  # 5% of the data‐range
    xmin, xmax = all_x.min() - margin, all_x.max() + margin
    ymin, ymax = all_y.min() - margin, all_y.max() + margin
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Random Deployment (all points guaranteed visible)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    # === 4) Save to file ===
    plt.savefig('deployment.png', dpi=150, bbox_inches='tight')
    plt.show()


def norm2_power2(rayleigh_faded_channel_matrix):
    return rayleigh_faded_channel_matrix[:, 0] ** 2 + rayleigh_faded_channel_matrix[:, 1] ** 2


def calculate_transmission_rate(sender, receiver, transmit_power, bandwidth=1000):  # bandwidth is Khz
    N0 = -174  # -174dBm/hz

    # Calculate transmitter to receiver channel gain------------
    channels = calculate_complex_channel_gain(sender, receiver, global_random_gn.standard_normal((1, 2)))
    transmit_powers = norm2_power2(channels) * transmit_power

    # calculate noise -----------------------------------------
    # bandwidth is khz, we convert it to hz via multiplying it by 1000
    # N0 is dBm/hz, we convert it to mW/hz
    # n0 is mW
    n0 = (10 ** (N0 / 10)) * bandwidth

    # Calculate SNIR
    SNIRs = transmit_powers / (n0)

    # Calculate data rate
    data_rate = bandwidth * np.log2(1 + SNIRs)  # kb/s
    # data_rates *= Environment.bit_per_hertz  # kbps
    # data_rates *= Environment.virtual_clock.time_unit  # kb/ms
    return data_rate[0]


def calculate_complex_channel_gain(sender, receiver, complex_fading_matrix):
    beta_0 = -30  # -30db
    d_0 = 1  # 1m
    alpha = 3
    rayleigh_faded_channel_matrix = complex_fading_matrix

    transmitter_receiver_distance = math.sqrt(np.power(receiver[0] - sender[0], 2)
                                              + np.power(receiver[1] - sender[1], 2))

    # transmit power in db
    clear_transmit_power = beta_0 - 10 * alpha * np.log10(transmitter_receiver_distance / d_0)
    # convert to watt
    clear_transmit_power = np.sqrt(10 ** (clear_transmit_power / 10))
    # applying rayleigh fading
    rayleigh_faded_channel_matrix *= clear_transmit_power

    return rayleigh_faded_channel_matrix


def main():
    # === PARAMETERS ===
    transmit_power = 200  # Mw
    n_servers = 10
    n_users = 35
    min_dist = 500.0  # minimum edge‐server separation (m)
    area_size = 5000.0  # make sure area is large enough to fit 20 servers ≥1000m apart
    model_size = 6 * 8000  # 1 MB = 8000 kbit

    # --- 1) Generate edge‐server positions with minimum‐distance constraint ---
    servers_positions = []
    while len(servers_positions) < n_servers:
        x, y = global_random_gn.uniform(0, area_size, size=2)
        if all(np.hypot(x - sx, y - sy) >= min_dist for sx, sy in servers_positions):
            servers_positions.append((x, y))
    servers_positions = np.array(servers_positions)

    user_max_dist_to_server = 1000.0
    user_cluster_radius = 800.0
    users_per_cluster = 5

    users_positions = []
    clusters = {cid: [] for cid in range(n_servers)}
    uid = -1

    for sid, (sx, sy) in enumerate(servers_positions):
        # pick a cluster center within 500 m of the server
        while True:
            r_center = np.sqrt(global_random_gn.random()) * user_max_dist_to_server
            theta_center = global_random_gn.random() * 2 * np.pi
            cx = sx + r_center * np.cos(theta_center)
            cy = sy + r_center * np.sin(theta_center)
            if 0 <= cx <= area_size and 0 <= cy <= area_size:
                break

        # generate users tightly clustered around (cx, cy)
        for _ in range(users_per_cluster):
            uid += 1
            while True:
                r_user = np.sqrt(global_random_gn.random()) * user_cluster_radius
                theta_user = global_random_gn.random() * 2 * np.pi
                ux = cx + r_user * np.cos(theta_user)
                uy = cy + r_user * np.sin(theta_user)
                if 0 <= ux <= area_size and 0 <= uy <= area_size:
                    users_positions.append((ux, uy))
                    clusters[sid].append(uid)
                    break

    users_positions = np.array(users_positions)
    path = 'clusters.npy'
    np.save(path, clusters)
    # clusters = np.array(clusters)

    # === 3) Plot ===
    draw_network(servers_positions, users_positions, area_size)

    up_data_rates = []
    for cid, users in clusters.items():
        for uid in users:
            up_data_rates.append(calculate_transmission_rate(users_positions[uid], servers_positions[cid], 1000))
            ux, uy = users_positions[uid]
            sx, sy = servers_positions[cid]
            print(
                f"Data rate: {up_data_rates[uid]:.2f} --> User {uid} at ({ux:.0f}, {uy:.0f}) → Server {cid} at ({sx:.0f}, {sy:.0f})")

    cluster_pair_rates = {cid: {} for cid in clusters}
    cluster_pair_latencies = {cid: {} for cid in clusters}
    for cid, uids in clusters.items():
        for i, uid1 in enumerate(uids):
            for uid2 in uids[i + 1:]:
                # positions of the two users
                p1 = users_positions[uid1]
                p2 = users_positions[uid2]
                # compute the transmission rate between them
                rate = calculate_transmission_rate(p1, p2, transmit_power, bandwidth=1000)
                # store it keyed by the (uid1, uid2) tuple
                cluster_pair_rates[cid][(uid1, uid2)] = rate
                cluster_pair_latencies[cid][(uid1, uid2)] = model_size / rate

    # --- example: print them out ---
    for cid, rates in cluster_pair_rates.items():
        print(f"\nCluster {cid} pairwise data‐rates (kb/s):")
        for (u1, u2), r in rates.items():
            print(f"  Users {u1}↔{u2}: {r:.2f}")

    up_latencies = {uid: 0 for uid in range(n_users)}
    cloud_latencies = {uid: 0 for uid in range(n_users)}
    computation_time = {uid: 0 for uid in range(n_users)}
    min_batch = 8
    local_sgd = 250
    frequency = 5.0
    a = 0.1
    server_to_cloud_latencies = {cid: 0 for cid in clusters.keys()}

    for cid, users in clusters.items():
        temp = 0
        for uid in users:
            latency = model_size / up_data_rates[uid]
            up_latencies[uid] = latency
            # cloud_latencies[uid] = 10* latency
            cloud_latencies[uid] = 0.1 + model_size / 4000
            computation_time[uid] = 18  # Real value from the server
            temp += 0.1 + model_size / 4000

        temp /= len(users)
        server_to_cloud_latencies[cid] = temp

    # Scenario 1
    epochs = 50
    gaf = 1
    cloud_transmit_power = 400
    local_transmit_power = 100
    effective_chipset = 2 * (10 ** (-27))

    avg_fedavg_latencies = {ep: max([computation_time[uid] for cid, users in clusters.items() for uid in users]) for ep
                            in range(epochs)}
    avg_fedavg_energies = {ep: sum(
        [effective_chipset / 2 * ((frequency * (10 ** 9)) ** 3) * computation_time[uid] for cid, users in
         clusters.items() for uid in users]) for ep in range(epochs)}

    # Scenario 2
    local_latencies = {cid: sum([computation_time[uid] + up_latencies[uid] * gaf for uid in users]) for cid, users in
                       clusters.items()}
    avg_hierarchy_latencies_2_4 = {
        ep: max([(server_to_cloud_latencies[cid] + local_latencies[cid]) / gaf for cid, users in clusters.items()]) for
        ep in range(epochs)}

    local_energies = {cid: sum([effective_chipset / 2 * ((frequency * (10 ** 9)) ** 3) * computation_time[uid] +
                                up_latencies[uid] * gaf * local_transmit_power for uid in users]) for cid, users in
                      clusters.items()}
    avg_hierarchy_energy_2_4 = {ep: sum(
        [(server_to_cloud_latencies[cid] * cloud_transmit_power + local_energies[cid]) / gaf for cid, users in
         clusters.items()]) for ep in range(epochs)}

    # Scenario 3
    cluster_pair_latencies = {cid: {} for cid in clusters}
    for cid, uids in clusters.items():
        for u1 in uids:
            for u2 in uids:
                if u1 == u2:
                    continue
                p1, p2 = users_positions[u1], users_positions[u2]
                rate = calculate_transmission_rate(p1, p2, transmit_power, bandwidth=8000)  # kb/s
                lat = model_size / rate  # s
                cluster_pair_latencies[cid][(u1, u2)] = lat

    # connectivity threshold
    conn_radius = 30

    relay_latencies_up = {}
    relay_latencies_down = {}
    relay_energies_up = {}
    relay_energies_down = {}
    d2d_local_transmit_power = 100
    for cid, uids in clusters.items():
        agg = random.choice(uids)
        print(f"\n=== Cluster {cid}: Aggregator = User {agg} ===")

        # 1) Build directed geometric graph
        G = nx.DiGraph()
        G.add_nodes_from(uids)
        for u1, u2 in combinations(uids, 2):
            p1, p2 = users_positions[u1], users_positions[u2]
            if np.linalg.norm(p1 - p2) <= conn_radius:
                # if within range, add both directed edges
                G.add_edge(u1, u2, weight=cluster_pair_latencies[cid][(u1, u2)])
                G.add_edge(u2, u1, weight=cluster_pair_latencies[cid][(u2, u1)])

        # 2) Ensure strong connectivity by patching via undirected MST
        if not nx.is_strongly_connected(G):
            # build an undirected surrogate with min(lat,lat_rev)
            H = nx.Graph()
            H.add_nodes_from(uids)
            for (u1, u2), lat in cluster_pair_latencies[cid].items():
                lat_rev = cluster_pair_latencies[cid].get((u2, u1), np.inf)
                H.add_edge(u1, u2, weight=min(lat, lat_rev))
            T = nx.minimum_spanning_tree(H, weight='weight')
            for u1, u2, d in T.edges(data=True):
                if not G.has_edge(u1, u2):
                    G.add_edge(u1, u2, weight=cluster_pair_latencies[cid][(u1, u2)])
                if not G.has_edge(u2, u1):
                    G.add_edge(u2, u1, weight=cluster_pair_latencies[cid][(u2, u1)])

        # 4) Shortest paths & latencies: user → agg
        fpaths, flats = {}, {}
        for uid in uids:
            if uid == agg:
                fpaths[uid], flats[uid] = [agg], 0.0
            else:
                fpaths[uid] = nx.shortest_path(G, uid, agg, weight='weight')
                flats[uid] = nx.shortest_path_length(G, uid, agg, weight='weight')
            print(f"User {uid} → Agg {agg} | Path {fpaths[uid]} | Latency {flats[uid]:.3f}s")

        # 5) Energy to aggregate: sum P·t over all users → agg, convert to kJ
        energy_to_agg_kJ = {uid: d2d_local_transmit_power * flats[uid] for uid in uids}
        total_e2a = sum(energy_to_agg_kJ.values())
        print(f"\nCluster {cid}: Energy to aggregate = {total_e2a:.4f} kJ")

        # 6) Shortest paths: agg → each user (directed)
        bpaths, blats = {}, {}
        for uid in uids:
            if uid == agg:
                bpaths[uid], blats[uid] = [agg], 0.0
            else:
                bpaths[uid] = nx.shortest_path(G, agg, uid, weight='weight')
                blats[uid] = nx.shortest_path_length(G, agg, uid, weight='weight')
            print(f"Agg {agg} → User {uid} | Path {bpaths[uid]} | Latency {blats[uid]:.3f}s")

        # 7) Compute broadcast tree edges (unique) and energy
        broadcast_edges = set()
        for path in bpaths.values():
            for i in range(len(path) - 1):
                broadcast_edges.add((path[i], path[i + 1]))

        energy_bcast_kJ = 0.0
        for u, v in broadcast_edges:
            lat = G[u][v]['weight']
            e = d2d_local_transmit_power * lat
            energy_bcast_kJ += e
            print(f"  Edge {u}→{v}: Lat {lat:.3f}s → E {e:.6f} kJ")

        bcast_time = max(blats.values())
        print(f"\nCluster {cid}: Broadcast time = {bcast_time:.3f}s")
        print(f"Cluster {cid}: Energy to broadcast = {energy_bcast_kJ:.4f} kJ")

        # 8) Total round‑trip energy
        print(f"Cluster {cid}: Total RT energy = {total_e2a + energy_bcast_kJ:.4f} kJ\n")

        relay_latencies_up[cid] = max(flats.values())
        relay_latencies_down[cid] = bcast_time
        relay_energies_up[cid] = total_e2a
        relay_energies_down[cid] = energy_bcast_kJ

    gaf = 1
    local_latencies = {cid: np.mean([computation_time[uid] + up_latencies[uid] for uid in users]) for cid, users in
                       clusters.items()}
    avg_relay_latencies_2_4 = {ep: max([(server_to_cloud_latencies[cid] + (
            relay_latencies_up[cid] + relay_latencies_down[cid]) * (gaf - 1) + relay_latencies_up[cid] +
                                         local_latencies[cid]) / gaf
                                        for cid, users in clusters.items()]) for ep in range(epochs)}

    avg_relay_energy_2_4 = {ep: sum([effective_chipset / 2 * ((frequency * (10 ** 9)) ** 3) * computation_time[uid] + (
            server_to_cloud_latencies[cid] * cloud_transmit_power + (
            relay_energies_up[cid] + relay_energies_down[cid]) * (gaf - 1) + relay_energies_up[cid] +
            local_latencies[cid] * local_transmit_power) / gaf for cid, users in clusters.items()]) for ep in
                            range(epochs)}

    import os

    def load_metrics_df(folder_name):
        path = os.path.join("results", folder_name, "accuracy_record.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return pd.read_json(path)

    df_fedavg = load_metrics_df("local")
    df_hierarchy = load_metrics_df("ffm")

    def mean_from_epoch_dict(d):
        return np.mean(list(d.values()))

    # Last common epoch
    common_epochs = set(df_fedavg["epoch"]).intersection(
        df_hierarchy["epoch"]
    )
    last_common_epoch = max(common_epochs)

    loss_fedavg_last = df_fedavg[df_fedavg["epoch"] == last_common_epoch]["accuracy"].mean()
    loss_hierarchy_last = df_hierarchy[df_hierarchy["epoch"] == last_common_epoch]["accuracy"].mean()
    loss_relay_last = loss_hierarchy_last


    results = {
        "Method": ["local", "ffm", "Relay"],
        "Avg Latency (s)": [
            mean_from_epoch_dict(avg_fedavg_latencies),
            mean_from_epoch_dict(avg_hierarchy_latencies_2_4),
            mean_from_epoch_dict(avg_relay_latencies_2_4),
        ],
        "Avg Energy (J)": [
            mean_from_epoch_dict(avg_fedavg_energies) / 1000,
            mean_from_epoch_dict(avg_hierarchy_energy_2_4) / 1000,
            mean_from_epoch_dict(avg_relay_energy_2_4) / 1000,
        ],
        "Loss at Last Epoch": [
            loss_fedavg_last,
            loss_hierarchy_last,
            loss_relay_last,
        ]
    }

    df = pd.DataFrame(results)
    print(df.to_latex(
        index=False,
        float_format="%.4f",
        caption=f"Comparison at Last Common Epoch {last_common_epoch}",
        label="tab:comparison"
    ))

    # Last common epoch
    common_epochs = set(df_fedavg["epoch"]).intersection(df_hierarchy["epoch"])
    last_common_epoch = max(common_epochs)

    # Mean loss across all users/tasks at that epoch
    loss_fedavg_last = df_fedavg[df_fedavg["epoch"] == last_common_epoch]["loss"].mean()
    loss_hierarchy_last = df_hierarchy[df_hierarchy["epoch"] == last_common_epoch]["loss"].mean()
    loss_relay_last = loss_hierarchy_last  # relay reuses hierarchy_2_4 metrics

    def compute_mean_metrics_per_epoch(df):
        # Step 1: Average over tasks for each (epoch, user)
        user_level = df.groupby(['epoch', 'user'])[['loss', 'accuracy']].mean().reset_index()
        # Step 2: Average over users for each epoch
        epoch_level = user_level.groupby('epoch')[['loss', 'accuracy']].mean().reset_index()
        return epoch_level

    def threshold_mask(x_cum, threshold):
        return x_cum <= threshold

    def compute_mean_std_metrics_per_epoch(df):
        user_level = df.groupby(['epoch', 'user'])[['loss', 'accuracy']].mean().reset_index()
        epoch_level = user_level.groupby('epoch')['accuracy'].agg(['mean', 'std']).reset_index()
        return epoch_level

    def plot_trend(x, y, color, label, marker='o', linestyle='-', th=34000, y_std=None, offset_x=1000, offset_y=5,
                   ax=None):
        x_cum = x
        mask = threshold_mask(x_cum, th)
        n = np.sum(mask)
        n2 = min(len(x), len(y))
        n = min(n, n2)
        x_c = x_cum[:n]
        y_c = y[:n]

        ax.plot(x_c, y_c, marker=marker, linestyle=linestyle, color=color, label=label, linewidth=2, markersize=6,
                markevery=3)

        # Add shaded std region if provided
        if y_std is not None:
            y_std_c = y_std[:n]
            ax.fill_between(x_c, y_c - y_std_c, y_c + y_std_c, color=color, alpha=0.1)

        x_last = x_c[-1]
        y_last = y_c[-1]
        ax.annotate(r"$\mathbf{Convergence\ Point\ (\pm 0.5\%)}$",
                    xy=(x_last, y_last),
                    xytext=(x_last - offset_x, y_last - offset_y),  # tweak offset as needed
                    arrowprops=dict(arrowstyle='->', color=color),
                    fontsize=12, color=color)

    def cumulative(x):
        """Return the cumulative sum of x."""
        return np.cumsum(x)

    colors = {
        "FedAvg": "#0f258f",
        "Hierarchy 2-4": "red",
        "Relay 2-4": "green",
    }

    x_fedavg = list(avg_fedavg_latencies.values())
    x_fedavg[0] = 0
    y_fedavg = compute_mean_metrics_per_epoch(df_fedavg)["accuracy"].values

    stats_fedavg = compute_mean_std_metrics_per_epoch(df_fedavg)
    y_fedavg = stats_fedavg["mean"].values
    y_fedavg_std = stats_fedavg["std"].values

    x_h2_4 = list(avg_hierarchy_latencies_2_4.values())
    x_h2_4[0] = 0
    y_h2_4 = compute_mean_metrics_per_epoch(df_hierarchy)["accuracy"].values
    y_h2_4[0] = y_fedavg[0]

    stats_hierarchy = compute_mean_std_metrics_per_epoch(df_hierarchy)
    y_h2_4 = stats_hierarchy["mean"].values
    y_h2_4_std = stats_hierarchy["std"].values
    y_h2_4[0] = y_fedavg[0]

    x_r2_4 = list(avg_relay_latencies_2_4.values())
    x_r2_4[0] = 0
    y_r2_4 = compute_mean_metrics_per_epoch(df_hierarchy)["accuracy"].values
    y_r2_4[0] = y_fedavg[0]

    stats_hierarchy = compute_mean_std_metrics_per_epoch(df_hierarchy)
    y_r2_4 = stats_hierarchy["mean"].values
    y_r2_4_std = stats_hierarchy["std"].values
    y_r2_4[0] = y_fedavg[0]

    x_fedavg_c = cumulative(x_fedavg)
    x_h2_4_c = cumulative(x_h2_4)
    x_r2_4_c = cumulative(x_r2_4)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # --- Subplot (a): Accuracy vs Latency ---
    plot_trend(x_fedavg_c, y_fedavg, color=colors["FedAvg"], marker='o', label='Local',
               th=4800, y_std=y_fedavg_std, offset_x=400, ax=axes[1])
    plot_trend(x_h2_4_c, y_h2_4, color=colors["Hierarchy 2-4"], marker='v', label='FFM',
               th=4800, y_std=y_h2_4_std, offset_x=1500, ax=axes[1])
    plot_trend(x_r2_4_c, y_r2_4, color=colors["Relay 2-4"], marker='s', label='FFM + P2P Relay',
               th=4800, y_std=y_r2_4_std, offset_x=400, offset_y=-5, ax=axes[1])

    axes[1].set_xlabel("Overall Training Latency (s)")
    axes[1].grid(True, which='major', alpha=0.6)
    axes[1].grid(True, which='minor', linestyle='--', alpha=0.3)

    x_fedavg = np.array(list(avg_fedavg_energies.values())) / 1000
    x_fedavg[0] = 0
    y_fedavg = compute_mean_metrics_per_epoch(df_fedavg)["accuracy"].values

    x_h2_4 = np.array(list(avg_hierarchy_energy_2_4.values())) / 1000
    x_h2_4[0] = 0
    y_h2_4 = compute_mean_metrics_per_epoch(df_hierarchy)["accuracy"].values
    y_h2_4[0] = y_fedavg[0]

    stats_hierarchy = compute_mean_std_metrics_per_epoch(df_hierarchy)
    y_h2_4 = stats_hierarchy["mean"].values
    y_h2_4_std = stats_hierarchy["std"].values
    y_h2_4[0] = y_fedavg[0]

    x_r2_4 = np.array(list(avg_relay_energy_2_4.values())) / 1000
    x_r2_4[0] = 0
    y_r2_4 = compute_mean_metrics_per_epoch(df_hierarchy)["accuracy"].values
    y_r2_4[0] = y_fedavg[0]

    stats_hierarchy = compute_mean_std_metrics_per_epoch(df_hierarchy)
    y_r2_4 = stats_hierarchy["mean"].values
    y_r2_4_std = stats_hierarchy["std"].values
    y_r2_4[0] = y_fedavg[0]

    x_fedavg_c = cumulative(x_fedavg)
    x_h2_4_c = cumulative(x_h2_4)
    x_r2_4_c = cumulative(x_r2_4)

    # --- Subplot (b): Accuracy vs Energy ---
    plot_trend(x_fedavg_c, y_fedavg, color=colors["FedAvg"], marker='o', label='Local',
               y_std=y_fedavg_std, ax=axes[0])
    plot_trend(x_h2_4_c, y_h2_4, color=colors["Hierarchy 2-4"], marker='v', label='FFM',
               y_std=y_h2_4_std, offset_x=2600, ax=axes[0])
    plot_trend(x_r2_4_c, y_r2_4, color=colors["Relay 2-4"], marker='s', label='FFM + P2P Relay',
               y_std=y_r2_4_std, offset_y=-5, ax=axes[0])

    axes[0].set_xlabel("Cumulative Energy Consumption Across Users (kJ)")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(True, which='major', alpha=0.6)
    axes[0].grid(True, which='minor', linestyle='--', alpha=0.3)

    # --- Shared Legend at Top ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=3,
               fontsize=15,
               frameon=True,
               fancybox=True,
               edgecolor='black',
               labelspacing=0.6,
               columnspacing=1.5,
               handletextpad=0.6,
               bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.90], w_pad=-2)
    # Adjust layout
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for top legend
    plt.savefig("combined_accuracy_plots_shared_legend.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
