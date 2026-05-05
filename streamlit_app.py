
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# STREAMLIT CHANGE: removed ffmpeg import because it is not needed for browser display
# import ffmpeg

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

class InnovationAgent(Agent):
    """
    Agent in an innovation-adoption system.

    State:
        0 = Not adopted
        1 = Considering
        2 = Adopted
    """

    def __init__(self, model, node_id, agent_type):
        super().__init__(model)

        self.node_id = node_id
        self.agent_type = agent_type

        self.state = 0
        self.next_state = 0

        # Neutral attributes
        self.openness = self.random.uniform(0.2, 1.0)
        self.resistance = self.random.uniform(0.3, 1.0)
        self.influence = self.random.uniform(0.5, 1.5)

        # Modify attributes by type
        if self.agent_type == "Champion":
            self.openness *= 1.4
            self.resistance *= 0.6
            self.influence *= 1.6

        elif self.agent_type == "Skeptic":
            self.openness *= 0.7
            self.resistance *= 1.4

        elif self.agent_type == "Manager":
            self.influence *= 2.0

    def decide(self):
        """
        Compute next state based on neighbouring adopted agents.
        """

        # Already adopted agents stay adopted
        if self.state == 2:
            self.next_state = 2
            return

        neighbor_nodes = list(self.model.network.neighbors(self.node_id))
        neighbor_agents = [self.model.node_to_agent[n] for n in neighbor_nodes]

        if len(neighbor_agents) == 0:
            social_pressure = 0
        else:
            adopted_neighbors = [a for a in neighbor_agents if a.state == 2]
            social_pressure = sum(a.influence for a in adopted_neighbors) / len(neighbor_agents)

        adoption_score = (
            self.openness
            + social_pressure
            + self.model.external_support
        )

        if adoption_score > self.resistance + self.model.adoption_margin:
            self.next_state = 2

        elif adoption_score > self.resistance:
            self.next_state = 1

        else:
            self.next_state = self.state

    def advance(self):
        """
        Update state after all agents have made their decision.
        """
        self.state = self.next_state


class InnovationAdoptionModel(Model):
    """
    Mesa model for innovation adoption in a small networked system.
    """

    def __init__(
        self,
        N,
        p_connection,
        external_support,
        adoption_margin,
        # STREAMLIT CHANGE: added skepticism_ratio as model input
        skepticism_ratio,
        seed=7
    ):
        super().__init__(seed=seed)

        self.N = N
        self.p_connection = p_connection
        self.external_support = external_support
        self.adoption_margin = adoption_margin

        # STREAMLIT CHANGE: store browser input inside the model
        self.skepticism_ratio = skepticism_ratio

        self.current_step = 0

        # Create connected network
        self.network = nx.erdos_renyi_graph(
            self.N,
            self.p_connection,
            seed=seed
        )

        while not nx.is_connected(self.network):
            self.network = nx.erdos_renyi_graph(
                self.N,
                self.p_connection,
                seed=self.random.randint(0, 10000)
            )

        self.grid = NetworkGrid(self.network)

        # STREAMLIT CHANGE: agent-type weights now depend on skepticism_ratio
        champion_ratio = 0.10
        manager_ratio = 0.10
        neutral_ratio = 1 - champion_ratio - manager_ratio - skepticism_ratio

        # Agent types
        agent_types = self.random.choices(
            ["Champion", "Neutral", "Skeptic", "Manager"],
            # STREAMLIT CHANGE: fixed skeptic weight replaced by slider-controlled value
            weights=[champion_ratio, neutral_ratio, skepticism_ratio, manager_ratio],
            k=self.N
        )

        self.node_to_agent = {}

        # Create and place agents
        for node_id in self.network.nodes:
            agent = InnovationAgent(
                model=self,
                node_id=node_id,
                agent_type=agent_types[node_id]
            )

            self.grid.place_agent(agent, node_id)
            self.node_to_agent[node_id] = agent

        # Initial adopters: preferably champions
        champions = [
            agent for agent in self.node_to_agent.values()
            if agent.agent_type == "Champion"
        ]

        if len(champions) >= 2:
            initial_adopters = self.random.sample(champions, 2)
        elif len(champions) == 1:
            others = [
                agent for agent in self.node_to_agent.values()
                if agent.agent_type != "Champion"
            ]
            initial_adopters = champions + self.random.sample(others, 1)
        else:
            initial_adopters = self.random.sample(
                list(self.node_to_agent.values()),
                2
            )

        for agent in initial_adopters:
            agent.state = 2
            agent.next_state = 2

        self.datacollector = DataCollector(
            model_reporters={
                "Adopted": lambda m: sum(a.state == 2 for a in m.node_to_agent.values()),
                "Considering": lambda m: sum(a.state == 1 for a in m.node_to_agent.values()),
                "Not adopted": lambda m: sum(a.state == 0 for a in m.node_to_agent.values()),
            },
            agent_reporters={
                "State": "state",
                "Agent type": "agent_type",
                "Openness": "openness",
                "Resistance": "resistance",
                "Influence": "influence"
            }
        )

        self.datacollector.collect(self)

    def step(self):
        """
        One simulation step.
        """

        # First all agents decide based on current states
        self.agents.do("decide")

        # Then all agents update simultaneously
        self.agents.do("advance")

        self.current_step += 1
        self.datacollector.collect(self)


# ============================================================
# STREAMLIT APP SECTION
# Everything below replaces your normal notebook-running section.
# ============================================================

# STREAMLIT CHANGE: browser title
st.title("Innovation Adoption Model")

# STREAMLIT CHANGE: browser input slider
skepticism_ratio = st.slider(
    "Select skepticism ratio",
    min_value=0.1,
    max_value=0.7,
    value=0.2,
    step=0.1
)

# STREAMLIT CHANGE: display chosen input
st.write("Selected skepticism ratio:", skepticism_ratio)

# STREAMLIT CHANGE: model now receives skepticism_ratio from the browser
model = InnovationAdoptionModel(
    N=30,
    p_connection=0.12,
    external_support=0.02,
    adoption_margin=0.35,
    skepticism_ratio=skepticism_ratio,
    seed=7
)

T = 10

for _ in range(T):
    model.step()

model_data = model.datacollector.get_model_vars_dataframe()

# STREAMLIT CHANGE: show data table in browser
st.subheader("Model data")
st.dataframe(model_data)


# ============================================================
# First output: line plot
# ============================================================

# STREAMLIT CHANGE: use fig, ax instead of plt.figure()
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(model_data.index, model_data["Adopted"], marker="o", color="lightgreen", label="Adopted")
ax1.plot(model_data.index, model_data["Considering"], marker="s", color="orange", label="Considering")
ax1.plot(model_data.index, model_data["Not adopted"], marker="^", color="lightblue", label="Not adopted")

ax1.set_xlabel("Time step")
ax1.set_ylabel("Number of agents")
ax1.set_title("Innovation Adoption Over Time")
ax1.legend()
ax1.grid(True)

# STREAMLIT CHANGE: display plot in browser instead of plt.show()
st.subheader("Innovation Adoption Over Time")
st.pyplot(fig1)


# ============================================================
# Second output: final network plot
# ============================================================

# Fixed positions for all animation frames
pos = nx.spring_layout(model.network, seed=7)

state_color_map = {
    0: "lightblue",
    1: "orange",
    2: "lightgreen"
}

type_marker_map = {
    "Champion": "*",
    "Neutral": "o",
    "Skeptic": "s",
    "Manager": "D"
}

state_label_map = {
    0: "Not adopted",
    1: "Considering",
    2: "Adopted"
}

# Store state history manually from agent data
agent_data = model.datacollector.get_agent_vars_dataframe()

final_step = model_data.index.max()
final_agent_data = agent_data.loc[final_step]

# STREAMLIT CHANGE: use fig, ax instead of plt.figure()
fig2, ax2 = plt.subplots(figsize=(9, 7))

for agent_type, marker in type_marker_map.items():
    nodes_of_type = [
        agent.node_id
        for agent in model.node_to_agent.values()
        if agent.agent_type == agent_type
    ]

    colors = [
        state_color_map[model.node_to_agent[node].state]
        for node in nodes_of_type
    ]

    nx.draw_networkx_nodes(
        model.network,
        pos,
        nodelist=nodes_of_type,
        node_color=colors,
        node_shape=marker,
        node_size=600,
        edgecolors="black",
        linewidths=1.0,
        label=agent_type,
        # STREAMLIT CHANGE: draw on ax2
        ax=ax2
    )

nx.draw_networkx_edges(
    model.network,
    pos,
    edge_color="gray",
    alpha=0.5,
    # STREAMLIT CHANGE: draw on ax2
    ax=ax2
)

nx.draw_networkx_labels(
    model.network,
    pos,
    font_size=9,
    # STREAMLIT CHANGE: draw on ax2
    ax=ax2
)

ax2.set_title("Final Innovation Adoption State")
ax2.axis("off")

state_legend = [
    Line2D([0], [0], marker="o", color="w", label="Adopted",
           markerfacecolor="lightgreen", markeredgecolor="black", markersize=10),
    Line2D([0], [0], marker="o", color="w", label="Considering",
           markerfacecolor="orange", markeredgecolor="black", markersize=10),
    Line2D([0], [0], marker="o", color="w", label="Not adopted",
           markerfacecolor="lightblue", markeredgecolor="black", markersize=10),
]

ax2.legend(handles=state_legend, loc="upper right")

# STREAMLIT CHANGE: display final network in browser instead of plt.show()
st.subheader("Final Innovation Adoption Network")
st.pyplot(fig2)