import pandas as pd

from context_engine import calculate_priority
from scheduler import compare_schedulers


DATASET_PATH = "data/processed/traffic_dataset.csv"


def create_context_flows(sample_size=100):
    """
    Create a representative set of flows from the
    actual extracted traffic dataset.
    """

    df = pd.read_csv(DATASET_PATH)

    # Take a reproducible sample
    sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    ).reset_index(drop=True)

    results = []

    # Simulated current network conditions
    network_load = 0.80
    queue_condition = 0.70

    for index, flow in sample.iterrows():

        context = calculate_priority(
            traffic_class=flow["traffic_class"],
            network_load=network_load,
            queue_condition=queue_condition
        )

        results.append({
            "flow_id": index + 1,
            "traffic_class": flow["traffic_class"],
            "total_bytes": flow["total_bytes"],
            "packet_count": flow["packet_count"],
            "priority_score": context["priority_score"],
            "priority_level": context["priority_level"]
        })

    return pd.DataFrame(results)


def run_simulation():

    print("\nLoading real flow dataset...")

    flows = create_context_flows(sample_size=100)

    print(f"Flows selected: {len(flows)}")

    print("\nTraffic distribution:")
    print(flows["traffic_class"].value_counts())

    print("\nPriority distribution:")
    print(flows["priority_level"].value_counts())

    print("\nFirst 10 context-aware flows:")
    print(
        flows[
            [
                "flow_id",
                "traffic_class",
                "priority_score",
                "priority_level",
                "total_bytes"
            ]
        ].head(10)
    )

    # Compare scheduling strategies
    fifo, context, comparison = compare_schedulers(
        flows
    )

    print("\n" + "=" * 60)
    print("REAL FLOW SCHEDULING COMPARISON")
    print("=" * 60)

    print(comparison)

    # Save results
    fifo.to_csv(
        "data/processed/fifo_results.csv",
        index=False
    )

    context.to_csv(
        "data/processed/context_results.csv",
        index=False
    )

    comparison.to_csv(
        "data/processed/scheduler_comparison.csv",
        index=False
    )

    print("\nResults saved:")
    print("data/processed/fifo_results.csv")
    print("data/processed/context_results.csv")
    print("data/processed/scheduler_comparison.csv")


if __name__ == "__main__":
    run_simulation()