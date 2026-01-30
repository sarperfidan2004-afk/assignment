import simpy
import random
import numpy as np

RANDOM_SEED = 42
SIM_TIME = 480  
random.seed(RANDOM_SEED)

wait_times = []
abandoned = 0
served = 0
busy_time = 0

def customer(env, name, agents, service_time, abandon_time):
    global abandoned, served, busy_time

    arrival_time = env.now

    with agents.request() as request:
        results = yield request | env.timeout(abandon_time)

        if request not in results:
            abandoned += 1
            return

        wait = env.now - arrival_time
        wait_times.append(wait)

        service_duration = random.expovariate(1.0 / service_time)
        busy_time += service_duration

        yield env.timeout(service_duration)
        served += 1

def arrival_process(env, channel_name, agents, arrival_rate, service_time, abandon_time):
    i = 0
    while True:
        yield env.timeout(random.expovariate(arrival_rate))
        env.process(customer(env, f"{channel_name}_{i}", agents, service_time, abandon_time))
        i += 1

def run_simulation(params):
    global wait_times, abandoned, served, busy_time

    wait_times = []
    abandoned = 0
    served = 0
    busy_time = 0

    env = simpy.Environment()

    phone_agents = simpy.Resource(env, capacity=params["phone_agents"])
    email_agents = simpy.Resource(env, capacity=params["email_agents"])
    chat_agents = simpy.Resource(env, capacity=params["chat_agents"])

    env.process(arrival_process(env, "Phone", phone_agents,
                                 params["phone_arrival"], params["phone_service"], params["phone_abandon"]))

    env.process(arrival_process(env, "Email", email_agents,
                                 params["email_arrival"], params["email_service"], params["email_abandon"]))

    env.process(arrival_process(env, "Chat", chat_agents,
                                 params["chat_arrival"], params["chat_service"], params["chat_abandon"]))

    env.run(until=SIM_TIME)

    avg_wait = np.mean(wait_times) if wait_times else 0
    abandonment_rate = abandoned / (served + abandoned) * 100 if (served + abandoned) > 0 else 0
    utilisation = busy_time / (SIM_TIME * (params["phone_agents"] +
                                           params["email_agents"] +
                                           params["chat_agents"])) * 100

    return avg_wait, abandonment_rate, utilisation

baseline = {
    "phone_arrival": 30/60, 
    "email_arrival": 15/60,
    "chat_arrival": 10/60,

    "phone_service": 5,
    "email_service": 8,
    "chat_service": 6,

    "phone_agents": 5,
    "email_agents": 4,
    "chat_agents": 3,

    "phone_abandon": 10,
    "email_abandon": 15,
    "chat_abandon": 12
}

peak = baseline.copy()
peak["phone_arrival"] = 40/60
peak["email_arrival"] = 30/60
peak["chat_arrival"] = 20/60

improvement = baseline.copy()
improvement["phone_agents"] = 6
improvement["email_agents"] = 5
improvement["chat_agents"] = 4

scenarios = {"Baseline": baseline, "Peak": peak, "Improvement": improvement}

for name, params in scenarios.items():
    avg_wait, abandon_rate, utilisation = run_simulation(params)
    print(f"\n{name} Scenario")
    print(f"Average Waiting Time: {avg_wait:.2f} minutes")
    print(f"Abandonment Rate: {abandon_rate:.2f}%")
    print(f"Agent Utilisation: {utilisation:.2f}%")

import matplotlib.pyplot as plt

results = {
    "Scenario": [],
    "Avg Wait": [],
    "Abandonment": [],
    "Utilisation": []
}

for name, params in scenarios.items():
    avg_wait, abandon_rate, utilisation = run_simulation(params)

    print(f"\n{name} Scenario")
    print(f"Average Waiting Time: {avg_wait:.2f} minutes")
    print(f"Abandonment Rate: {abandon_rate:.2f}%")
    print(f"Agent Utilisation: {utilisation:.2f}%")

    results["Scenario"].append(name)
    results["Avg Wait"].append(avg_wait)
    results["Abandonment"].append(abandon_rate)
    results["Utilisation"].append(utilisation)

# ---- Plotting ----
x = np.arange(len(results["Scenario"]))
width = 0.25

plt.figure(figsize=(10,6))

plt.bar(x - width, results["Avg Wait"], width, label="Avg Waiting Time")
plt.bar(x, results["Abandonment"], width, label="Abandonment Rate (%)")
plt.bar(x + width, results["Utilisation"], width, label="Agent Utilisation (%)")

plt.xticks(x, results["Scenario"])
plt.ylabel("Value")
plt.title("Simulation Results by Scenario")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
