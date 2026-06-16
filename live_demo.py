from slice_selector import select_slice

print("\n===== LIVE 5G NETWORK DEMO =====")

tp = float(input("Enter Throughput (Mbps): "))
delay = float(input("Enter Delay (ms): "))
plr = float(input("Enter Packet Loss Rate: "))

selected = select_slice(tp, delay, plr)

print("\nAI Selected Slice:", selected)

if selected == "URLLC":

    print("\nGenAI Recommendation:")
    print("Use URLLC Slice")

    print("\nBest For:")
    print("- Cloud Gaming")
    print("- Video Calls")
    print("- Remote Control")

elif selected == "eMBB":

    print("\nGenAI Recommendation:")
    print("Use eMBB Slice")

    print("\nBest For:")
    print("- Netflix")
    print("- YouTube")
    print("- AR/VR")

else:

    print("\nGenAI Recommendation:")
    print("Use mMTC Slice")

    print("\nBest For:")
    print("- IoT Devices")
    print("- Sensors")
    print("- Smart Cities")