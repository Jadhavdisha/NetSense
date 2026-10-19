from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd


def extract_packet_features(pcap_file):
    """
    Extract packet-level features from a PCAP file.
    """

    packets = rdpcap(pcap_file)

    rows = []

    for packet in packets:

        # We only process IPv4 packets
        if IP not in packet:
            continue

        row = {
            "timestamp": float(packet.time),
            "src_ip": packet[IP].src,
            "dst_ip": packet[IP].dst,
            "protocol": packet[IP].proto,
            "packet_length": len(packet)
        }

        # TCP packet
        if TCP in packet:

            row["src_port"] = packet[TCP].sport
            row["dst_port"] = packet[TCP].dport
            row["transport"] = "TCP"

        # UDP packet
        elif UDP in packet:

            row["src_port"] = packet[UDP].sport
            row["dst_port"] = packet[UDP].dport
            row["transport"] = "UDP"

        # Other IP packet
        else:

            row["src_port"] = 0
            row["dst_port"] = 0
            row["transport"] = "OTHER"

        rows.append(row)

    return pd.DataFrame(rows)

def create_flow_features(packet_df):
    """
    Convert packet-level data into bidirectional flow-level features.
    """

    if packet_df.empty:
        return pd.DataFrame()

    # Create a bidirectional flow key.
    # The two endpoints are sorted so A->B and B->A
    # belong to the same flow.
    def create_flow_key(row):

        endpoint_1 = (
            str(row["src_ip"]),
            int(row["src_port"])
        )

        endpoint_2 = (
            str(row["dst_ip"]),
            int(row["dst_port"])
        )

        if endpoint_1 <= endpoint_2:
            first = endpoint_1
            second = endpoint_2
        else:
            first = endpoint_2
            second = endpoint_1

        return (
            first[0],
            first[1],
            second[0],
            second[1],
            int(row["protocol"])
        )

    packet_df = packet_df.copy()

    packet_df["flow_key"] = packet_df.apply(
        create_flow_key,
        axis=1
    )

    flows = []

    for flow_key, group in packet_df.groupby("flow_key"):

        group = group.sort_values("timestamp")

        start_time = group["timestamp"].min()
        end_time = group["timestamp"].max()

        duration = end_time - start_time

        packet_count = len(group)

        total_bytes = group["packet_length"].sum()

        average_packet_size = (
            total_bytes / packet_count
            if packet_count > 0
            else 0
        )

        packet_rate = (
            packet_count / duration
            if duration > 0
            else 0
        )

        byte_rate = (
            total_bytes / duration
            if duration > 0
            else 0
        )

        inter_arrival_times = (
            group["timestamp"].diff().dropna()
        )

        average_iat = (
            inter_arrival_times.mean()
            if len(inter_arrival_times) > 0
            else 0
        )

        flow = {
            "src_ip": flow_key[0],
            "src_port": flow_key[1],

            "dst_ip": flow_key[2],
            "dst_port": flow_key[3],

            "protocol": flow_key[4],

            "packet_count": packet_count,
            "total_bytes": total_bytes,
            "duration": duration,
            "average_packet_size": average_packet_size,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "average_iat": average_iat
        }

        flows.append(flow)

    return pd.DataFrame(flows)

if __name__ == "__main__":

    pcap_path = "data/raw/Gmail.pcap"

    print("Extracting packet features...")

    packet_df = extract_packet_features(pcap_path)

    print("\nPacket extraction completed!")
    print("Number of IPv4 packets:", len(packet_df))

    print("\nCreating flow features...")

    flow_df = create_flow_features(packet_df)

    print("\nFlow extraction completed!")
    print("Number of flows:", len(flow_df))

    print("\nFirst 5 flows:")
    print(flow_df.head())

    print("\nFlow columns:")
    print(flow_df.columns.tolist())

    output_file = "data/processed/gmail_flows.csv"

    flow_df.to_csv(output_file, index=False)

    print(f"\nFlow dataset saved to: {output_file}")
