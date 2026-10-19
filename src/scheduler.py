import pandas as pd


def fifo_schedule(flows):
    """
    Schedule flows using traditional First-In-First-Out.
    """

    scheduled = flows.copy().reset_index(drop=True)

    scheduled["schedule_order"] = range(1, len(scheduled) + 1)

    return scheduled


def context_aware_schedule(flows):
    """
    Schedule flows according to their context-aware priority.
    """

    scheduled = flows.copy()

    # Highest priority first
    scheduled = scheduled.sort_values(
        by="priority_score",
        ascending=False
    ).reset_index(drop=True)

    scheduled["schedule_order"] = range(
        1,
        len(scheduled) + 1
    )

    return scheduled


def calculate_waiting_time(scheduled):
    """
    Simulate waiting time based on service time.

    Service time is estimated from flow size.
    """

    scheduled = scheduled.copy()

    # Convert bytes into a simple simulated service time.
    # 1 MB/s service capacity.
    scheduled["service_time"] = (
        scheduled["total_bytes"] / 1_000_000
    )

    waiting_times = []

    current_time = 0.0

    for service_time in scheduled["service_time"]:

        waiting_times.append(current_time)

        current_time += service_time

    scheduled["waiting_time"] = waiting_times

    return scheduled


def calculate_metrics(scheduled):
    """Calculate performance metrics for a scheduling strategy."""

    total_bytes = scheduled["total_bytes"].sum()

    # Total simulated completion time
    total_time = scheduled["service_time"].sum()

    throughput = (
        total_bytes / total_time
        if total_time > 0
        else 0
    )

    high_priority = scheduled[
        scheduled["priority_score"] >= 0.70
    ]

    low_priority = scheduled[
        scheduled["priority_score"] < 0.70
    ]

    metrics = {
        "Average Waiting Time":
            scheduled["waiting_time"].mean(),

        "Maximum Waiting Time":
            scheduled["waiting_time"].max(),

        "High Priority Waiting Time":
            high_priority["waiting_time"].mean()
            if not high_priority.empty else 0,

        "Low Priority Waiting Time":
            low_priority["waiting_time"].mean()
            if not low_priority.empty else 0,

        "Throughput":
            throughput
    }

    return metrics


def compare_schedulers(flows):
    """
    Compare FIFO and context-aware scheduling.
    """

    fifo = fifo_schedule(flows)
    context = context_aware_schedule(flows)

    fifo = calculate_waiting_time(fifo)
    context = calculate_waiting_time(context)

    fifo_metrics = calculate_metrics(fifo)
    context_metrics = calculate_metrics(context)

    comparison = pd.DataFrame({
        "Metric": fifo_metrics.keys(),

        "FIFO": fifo_metrics.values(),

        "Context-Aware": context_metrics.values()
    })

    return fifo, context, comparison


if __name__ == "__main__":

    # Small demonstration
    demo_flows = pd.DataFrame({
        "flow_id": ["A", "B", "C", "D"],

        "traffic_class": [
            "Web",
            "Gaming",
            "Video/Communication",
            "Data Transfer"
        ],

        "priority_score": [
            0.31,
            0.87,
            0.68,
            0.45
        ],

        "total_bytes": [
            500_000,
            200_000,
            800_000,
            1_000_000
        ]
    })

    print("\nOriginal Flow Order")
    print("=" * 50)

    print(demo_flows)

    fifo, context, comparison = compare_schedulers(
        demo_flows
    )

    print("\nFIFO Scheduling")
    print("=" * 50)

    print(
        fifo[
            [
                "flow_id",
                "traffic_class",
                "priority_score",
                "schedule_order",
                "waiting_time"
            ]
        ]
    )

    print("\nContext-Aware Scheduling")
    print("=" * 50)

    print(
        context[
            [
                "flow_id",
                "traffic_class",
                "priority_score",
                "schedule_order",
                "waiting_time"
            ]
        ]
    )

    print("\nScheduler Comparison")
    print("=" * 50)

    print(comparison)