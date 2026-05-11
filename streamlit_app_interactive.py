
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit.components.v1 as components

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


#### Agents and attributes ####
class InnovationAgent(Agent):
   
    def __init__(self, model, node_id, agent_type):
        super().__init__(model)

        self.node_id = node_id
        self.agent_type = agent_type

        #### Adoption status
        self.state = 0
        self.next_state = 0

        #### Receptive people's attributes
        self.openness = self.random.uniform(0.3, 1.0)
        self.resistance = self.random.uniform(0.3, 0.9)
        self.influence = self.random.uniform(0.5, 1.5)

        #### Modify attributes by type
        if self.agent_type == "PolicyMaker":
            self.openness *= 1.4
            self.resistance *= 0.6
            self.influence *= 1.6

        elif self.agent_type == "Skeptic":
            self.openness *= 0.7
            self.resistance *= 1.4

        elif self.agent_type == "Manager":
            self.influence *= 2.0
            self.openness *= 1.3


    def decide(self):
        
        #### Already adopted agents stay adopted ####
        if self.state == 2:
            self.next_state = 2
            return
           
        #### Agents neighbors influence decision ####
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
        #### Next state
        self.state = self.next_state


class InnovationAdoptionModel(Model):
   
    def __init__(
        self,
        N,
        p_connection,
        external_support,
        adoption_margin,
        open_ratio,
        manager_ratio,
        seed=7
    ):
        super().__init__(seed=seed)

        self.N = N
        self.p_connection = p_connection
        self.external_support = external_support
        self.adoption_margin = adoption_margin

        #### Store browser input inside the model
        self.open_ratio = open_ratio
        self.manger_ratio = manager_ratio
        self.current_step = 0

        #### Create connected network
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

        policymaker_ratio = 0.04
        skeptic_ratio = 1 - policymaker_ratio - manager_ratio - open_ratio

        #### Agent types ratio
        agent_types = self.random.choices(
            ["PolicyMaker", "Open", "Skeptic", "Manager"],
            weights=[policymaker_ratio, open_ratio, skeptic_ratio, manager_ratio],
            k=self.N
        )

        self.node_to_agent = {}

        #### Create and place agents
        for node_id in self.network.nodes:
            agent = InnovationAgent(
                model=self,
                node_id=node_id,
                agent_type=agent_types[node_id]
            )

            self.grid.place_agent(agent, node_id)
            self.node_to_agent[node_id] = agent

        PolicyMakers = [
            agent for agent in self.node_to_agent.values()
            if agent.agent_type == "PolicyMaker"
        ]

        if len(PolicyMakers) >= 2:
            initial_adopters = self.random.sample(PolicyMakers, 2)
        elif len(PolicyMakers) == 1:
            others = [
                agent for agent in self.node_to_agent.values()
                if agent.agent_type != "PolicyMaker"
            ]
            initial_adopters = PolicyMakers + self.random.sample(others, 1)
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

        #### First all agents decide based on current states
        self.agents.do("decide")

        #### Then all agents update simultaneously
        self.agents.do("advance")

        self.current_step += 1
        self.datacollector.collect(self)



#### STREAMLIT APP SECTION

#### Browser title
st.title("Innovation Adoption Model")

#### Browser input
open_ratio = st.select_slider(
    "Select the ratio of people open to change",
    options = [0.08, 0.18, 0.26, 0.34, 0.4, 0.46, 0.5, 0.54, 0.58],
    value = 0.26
    )
manager_ratio = st.select_slider(
    "Select the ratio of persuasive managers",
    options = [0.02, 0.12, 0.2],
    value = 0.12
)

#### Initializing model
model = InnovationAdoptionModel(
    N=25,
    p_connection=0.12,
    external_support=0.02,
    adoption_margin=0.35,
    open_ratio=open_ratio,
    manager_ratio=manager_ratio,
    seed=7
)

#### Number of steps
T = 10

for _ in range(T):
    model.step()

model_data = model.datacollector.get_model_vars_dataframe()
list = ['state ' + str(i) for i in range(11)]
model_data.index=list

#### Show data table in browser
st.subheader("Model data")
st.dataframe(model_data)

#### Illustration on browser
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(model_data.index, model_data["Adopted"], marker="o", color="green", label="Adopted")
ax1.plot(model_data.index, model_data["Considering"], marker="s", color="orange", label="Considering")
ax1.plot(model_data.index, model_data["Not adopted"], marker="^", color="red", label="Not adopted")

ax1.set_xlabel("Time step")
ax1.set_ylabel("Number of agents")
ax1.set_title("Innovation Adoption Over Time")
ax1.legend()
ax1.grid(True)

st.subheader("Innovation Adoption Over Time")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize = (8,1.8))
fig2.subplots_adjust(right=0.80)

values = model_data["Adopted"].values
colors = plt.cm.Greens(np.clip(values / 25, 0, 1))
left = 0

for i, value in enumerate(values):
    ax2.barh(
        "Adopted",      # only one category, so one bar
        value,          # width of this segment
        left=left,      # stack to the right
        color=colors[i],
        edgecolor="white",
        height=0.35,
        label=f"Stage {i+1}"
    )
    left += value

norm = plt.Normalize(vmin=0, vmax=25)
sm = plt.cm.ScalarMappable(cmap="Greens", norm=norm)
sm.set_array([])

cax = fig2.add_axes([0.84, 0.15, 0.05, 0.7])
cb = fig2.colorbar(sm, cax=cax, orientation="vertical")

cb.set_ticks([0, 25])
cb.set_ticklabels(["min", "max"])

ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_xlabel("Expected Profit at each time step")
st.subheader("Expected profit with respect to number of adopters")
st.pyplot(fig2)

#### Fixed positions for all animation frames
pos = nx.spring_layout(model.network, seed=5)

state_color_map = {
    0: "red",
    1: "orange",
    2: "green"
}

type_marker_map = {
    "PolicyMaker": "*",
    "Open": "o",
    "Skeptic": "s",
    "Manager": "D"
}

state_label_map = {
    0: "Not adopted",
    1: "Considering",
    2: "Adopted"
}

#### Store state history
agent_data = model.datacollector.get_agent_vars_dataframe()

final_step = model_data.index.max()
final_agent_data = agent_data.loc[final_step]
st.write("**To see the different steps the ABM goes through please push the button**")

if st.button("Show all steps"):

   fig, ax = plt.subplots(figsize=(7, 7))

   def draw_frame(frame):
        ax.clear()

        frame_data = agent_data.loc[frame]

        for agent_type, marker in type_marker_map.items():
            nodes_of_type = []

            for node_id, agent in model.node_to_agent.items():
                if agent.agent_type == agent_type:
                    nodes_of_type.append(node_id)

            node_colors = []

            for node_id in nodes_of_type:
                # In Mesa 3, agent IDs are automatic.
                # We recover state by matching the agent objects' unique_id.
                agent = model.node_to_agent[node_id]
                state = frame_data.loc[agent.unique_id]["State"]
                node_colors.append(state_color_map[state])

            nx.draw_networkx_nodes(
                model.network,
                pos,
                nodelist=nodes_of_type,
                node_color=node_colors,
                node_shape=marker,
                node_size=750,
                edgecolors="black",
                linewidths=1.0,
                ax=ax
            )

        nx.draw_networkx_edges(
            model.network,
            pos,
            edge_color="gray",
            alpha=0.5,
            ax=ax
        )

        nx.draw_networkx_labels(
            model.network,
            pos,
            font_size=9,
            ax=ax
        )

        adopted = model_data.loc[frame, "Adopted"]
        considering = model_data.loc[frame, "Considering"]
        not_adopted = model_data.loc[frame, "Not adopted"]

        state_legend = [
            Line2D([0], [0], marker="o", color="w", label="Adopted",
                markerfacecolor="green", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Considering",
                markerfacecolor="orange", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Not adopted",
                markerfacecolor="red", markeredgecolor="black", markersize=10),
        ]

        ax.legend(handles=state_legend, loc="upper right")

   animation = FuncAnimation(
        fig,
        draw_frame,
        frames=list(model_data.index),
        interval=600,
        repeat=False
    )
   components.html(
    animation.to_jshtml(),
    height=1500,
    )
