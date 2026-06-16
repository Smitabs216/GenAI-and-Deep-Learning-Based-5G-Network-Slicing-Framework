import pandas as pd
import streamlit as st
from slice_selector import select_slice

# Page Configuration
st.set_page_config(page_title="5G Network Slicing Dashboard")

# Title
st.title(" AI-Driven 5G Network Slicing Dashboard")

st.write(
    "Enter live network parameters and let AI choose the best network slice."
)

# User Inputs
tp = st.number_input(
    "Throughput (Mbps)",
    min_value=0.0,
    value=10.0
)

delay = st.number_input(
    "Delay (ms)",
    min_value=0.0,
    value=20.0
)

plr = st.number_input(
    "Packet Loss Rate",
    min_value=0.0,
    value=0.00001,
    format="%.5f"
)

# Analyze Button
if st.button("Analyze Network"):

    # AI Slice Selection
    selected = select_slice(tp, delay, plr)

    # Network Health Score
    health = min(
        100,
        max(
            0,
            int(
                (tp / 5)
                + (100 - delay) * 0.7
            )
        )
    )

    st.metric(
        "Network Health Score",
        f"{health}/100"
    )

    # Selected Slice
    st.markdown(
        f"##  Selected Slice: {selected}"
    )

    # Slice Score Graph
    scores = {
        "eMBB": min(100, tp / 5),
        "URLLC": max(0, 100 - delay),
        "mMTC": max(0, 100 - (tp / 5))
    }

    df = pd.DataFrame(
        list(scores.items()),
        columns=["Slice", "Score"]
    )

    st.subheader("Slice Selection Scores")
    st.bar_chart(df.set_index("Slice"))

    # AI Explanation
    st.subheader("AI Explanation")

    if selected == "eMBB":

        st.info(
            "AI selected eMBB because the network offers high throughput suitable for video streaming, AR/VR, multimedia services, and high-bandwidth applications."
        )

        st.subheader("Recommended Applications")

        st.info("🎬 Netflix")
        st.info("📺 YouTube")
        st.info("🥽 AR/VR")
        st.info("📡 HD Video Streaming")

    elif selected == "URLLC":

        st.info(
            "AI selected URLLC because the network requires ultra-low latency and high reliability for real-time communication and mission-critical services."
        )

        st.subheader("Recommended Applications")

        st.info("🎮 Cloud Gaming")
        st.info("📞 Video Conferencing")
        st.info("🚗 Autonomous Vehicles")
        st.info("🏥 Remote Healthcare")

    else:

        st.info(
            "AI selected mMTC because the traffic pattern is suitable for massive IoT deployments with many connected devices and low bandwidth requirements."
        )

        st.subheader("Recommended Applications")

        st.info("🌾 Smart Agriculture")
        st.info("🏙 Smart Cities")
        st.info("📡 IoT Sensors")
        st.info("🏭 Industrial Monitoring")