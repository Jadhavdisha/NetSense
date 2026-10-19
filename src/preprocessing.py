import os
import pandas as pd

from feature_extraction import extract_packet_features, create_flow_features


# PCAP files and their traffic classes
PCAP_FILES = {
    "Gmail.pcap": "Web",
    "Skype.pcap": "Video/Communication",
    "WorldOfWarcraft.pcap": "Gaming",
    "BitTorrent.pcap": "Data Transfer"
}


def process_pcap(pcap_path, label):
    """Extract flow-level features from one PCAP and assign its class label."""

    print(f"\nProcessing: {pcap_path}")

    packet_df = extract_packet_features(pcap_path)

    if packet_df.empty:
        print("No valid IP packets found.")
        return pd.DataFrame()

    flow_df = create_flow_features(packet_df)

    flow_df["traffic_class"] = label
    flow_df["source_file"] = os.path.basename(pcap_path)

    print(f"Packets: {len(packet_df)}")
    print(f"Flows: {len(flow_df)}")
    print(f"Class: {label}")

    return flow_df


def build_dataset():
    all_flows = []

    for filename, label in PCAP_FILES.items():

        pcap_path = os.path.join("data", "raw", filename)

        if not os.path.exists(pcap_path):
            print(f"WARNING: File not found: {pcap_path}")
            continue

        flow_df = process_pcap(pcap_path, label)

        if not flow_df.empty:
            all_flows.append(flow_df)

    if not all_flows:
        print("\nERROR: No PCAP files were processed.")
        return

    dataset = pd.concat(all_flows, ignore_index=True)

    # Create output directory if needed
    os.makedirs("data/processed", exist_ok=True)

    output_file = "data/processed/traffic_dataset.csv"

    dataset.to_csv(output_file, index=False)

    print("\n" + "=" * 50)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 50)

    print("Shape:", dataset.shape)

    print("\nTraffic class distribution:")
    print(dataset["traffic_class"].value_counts())

    print("\nColumns:")
    print(dataset.columns.tolist())

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    build_dataset()