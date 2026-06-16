import matplotlib.pyplot as plt
from slice_selector import select_slice

print("\n========== NETWORK COMPARISON ==========\n")

print("NETWORK 1")
tp1 = float(input("Throughput (Mbps): "))
delay1 = float(input("Delay (ms): "))
plr1 = float(input("Packet Loss Rate: "))

print("\nNETWORK 2")
tp2 = float(input("Throughput (Mbps): "))
delay2 = float(input("Delay (ms): "))
plr2 = float(input("Packet Loss Rate: "))

slice1 = select_slice(tp1, delay1, plr1)
slice2 = select_slice(tp2, delay2, plr2)

print("\n===================================")
print("RESULTS")
print("===================================")

print(f"Network 1 Slice: {slice1}")
print(f"Network 2 Slice: {slice2}")
print("\n----- APPLICATION RECOMMENDATION -----")

# Streaming
# IoT
if tp1 < tp2:
    print("Best for IoT : Network 1")
else:
    print("Best for IoT : Network 2")

apps = ["Streaming","Gaming","IoT"]

net1 = [
    tp1,
    100-delay1,
    10
]

net2 = [
    tp2,
    100-delay2,
    10
]

x = range(len(apps))

plt.figure(figsize=(8,5))

plt.bar([i-0.2 for i in x], net1, width=0.4, label="Network 1")
plt.bar([i+0.2 for i in x], net2, width=0.4, label="Network 2")

plt.xticks(x, apps)
plt.ylabel("Score")
plt.title("Network Comparison")
plt.legend()

plt.show()