def generate_recommendation(demands):

    recommendations = {}

    for slice_name, d in demands.items():

        tp = d["Throughput_Mbps"]
        delay = d["Packet_Delay_Budget_ms"]
        plr = d["Packet_Loss_Rate"]

        if delay > 150:
            risk = "HIGH"

            recommendation = (
                "Increase Resource Blocks for this slice."
            )

        elif plr > 0.0001:
            risk = "MEDIUM"

            recommendation = (
                "Improve QoS configuration."
            )

        elif tp < 2:
            risk = "MEDIUM"

            recommendation = (
                "Allocate additional bandwidth."
            )

        else:
            risk = "LOW"

            recommendation = (
                "Current allocation is sufficient."
            )

        recommendations[slice_name] = {
            "risk": risk,
            "recommendation": recommendation
        }

    return recommendations