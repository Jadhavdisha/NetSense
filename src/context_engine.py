import pandas as pd


# Traffic-specific context
TRAFFIC_CONTEXT = {
    "Gaming": {
        "urgency": 0.95,
        "delay_sensitivity": 0.90
    },

    "Video/Communication": {
        "urgency": 0.80,
        "delay_sensitivity": 0.85
    },

    "Web": {
        "urgency": 0.50,
        "delay_sensitivity": 0.45
    },

    "Data Transfer": {
        "urgency": 0.30,
        "delay_sensitivity": 0.25
    }
}


def calculate_priority(
    traffic_class,
    network_load,
    queue_condition
):
    """
    Calculate context-aware priority for a network flow.

    All context values are normalized between 0 and 1.
    """

    if traffic_class not in TRAFFIC_CONTEXT:
        raise ValueError(
            f"Unknown traffic class: {traffic_class}"
        )

    urgency = TRAFFIC_CONTEXT[traffic_class]["urgency"]
    delay_sensitivity = TRAFFIC_CONTEXT[traffic_class][
        "delay_sensitivity"
    ]

    # Context-aware priority formula
    priority_score = (
        0.35 * urgency
        + 0.25 * delay_sensitivity
        + 0.20 * network_load
        + 0.20 * queue_condition
    )

    # Determine priority level
    if priority_score >= 0.70:
        priority_level = "HIGH"

    elif priority_score >= 0.40:
        priority_level = "MEDIUM"

    else:
        priority_level = "LOW"

    return {
        "traffic_class": traffic_class,
        "urgency": urgency,
        "delay_sensitivity": delay_sensitivity,
        "network_load": network_load,
        "queue_condition": queue_condition,
        "priority_score": priority_score,
        "priority_level": priority_level
    }


def analyze_flows(flow_df, network_load=0.70, queue_condition=0.60):
    """
    Apply context analysis to multiple flows.
    """

    results = []

    for _, flow in flow_df.iterrows():

        result = calculate_priority(
            traffic_class=flow["traffic_class"],
            network_load=network_load,
            queue_condition=queue_condition
        )

        result["source_file"] = flow.get(
            "source_file",
            "Unknown"
        )

        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":

    # Test the context engine
    test_classes = [
        "Gaming",
        "Video/Communication",
        "Web",
        "Data Transfer"
    ]

    print("\nContext Engine Test")
    print("=" * 60)

    for traffic_class in test_classes:

        result = calculate_priority(
            traffic_class=traffic_class,
            network_load=0.80,
            queue_condition=0.70
        )

        print(
            f"{traffic_class:25} "
            f"Score: {result['priority_score']:.3f} "
            f"Priority: {result['priority_level']}"
        )