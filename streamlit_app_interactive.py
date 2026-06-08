
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit.components.v1 as components
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import math

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
        if self.agent_type == "Policy Maker":
            self.openness *= 1.4
            self.resistance *= 0.6
            self.influence *= 1.6

        elif self.agent_type == "Neutral User":
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
           
        #### Agents' neighbors influence decision ####
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

        neutral_ratio = 1 - manager_ratio - open_ratio

        #### Agent types ratio — exactly 1 PolicyMaker, rest distributed by ratio
        agent_types = ["Policy Maker"] + self.random.choices(
            ["Receptive User", "Neutral User", "Manager"],
            weights=[open_ratio, neutral_ratio, manager_ratio],
            k=self.N - 1
        )
        self.random.shuffle(agent_types)

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
            if agent.agent_type == "Policy Maker"
        ]

        if len(PolicyMakers) >= 2:
            initial_adopters = self.random.sample(PolicyMakers, 2)
        elif len(PolicyMakers) == 1:
            others = [
                agent for agent in self.node_to_agent.values()
                if agent.agent_type != "Policy Maker"
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
st.title("Innovation Adoption Agent Based Model")
st.markdown("##")
st.write("The number of persuasive managers and people open to change is adjustable.")
# st.markdown("#")
#### Browser input
open_ratio = st.select_slider(
    "Select the ratio of people receptive to change",
    options = [0.08, 0.18, 0.26, 0.34, 0.4, 0.46, 0.5, 0.54, 0.58],
    value = 0.54
    )
manager_ratio = st.select_slider(
    "Select the ratio of persuasive managers",
    options = [0.02, 0.12, 0.2],
    value = 0.2
)

#### Initializing model
model = InnovationAdoptionModel(
    N=30,
    p_connection=0.15,
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
list_table = ['step ' + str(i) for i in range(T + 1)]
df_table = model_data.copy()
df_table.index = list_table

# #### Show data table in browser
# st.subheader("Model data at each step")
# st.dataframe(df_table)

#### Illustration on browser
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(model_data.index, model_data["Not adopted"], marker="o", color="#e74c3c", label="Not adopted")
ax1.plot(model_data.index, model_data["Considering"], marker="o", color="#f39c12", label="Considering")
ax1.plot(model_data.index, model_data["Adopted"], marker="o", color="#27ae60", label="Adopted")

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
pos = nx.spring_layout(model.network, seed=4)

state_color_map = {
    0: "#e63a27",
    1: "#f39c12",
    2: "#27ae60"
}

type_marker_map = {
    "Policy Maker": "P",
    "Receptive User": "o",
    "Neutral User": "s",
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
st.markdown("#")
st.subheader("**Different steps the ABM goes through**")
# st.button("Show all steps")

if 1==1:

   fig, ax = plt.subplots(figsize=(6, 5))
   fig.subplots_adjust(right=0.74)

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

            node_size_map_2d = {
                "Policy Maker": 550,
                "Receptive User": 550,
                "Neutral User": 550,
                "Manager": 550
            }

            nx.draw_networkx_nodes(
                model.network,
                pos,
                nodelist=nodes_of_type,
                node_color=node_colors,
                node_shape=marker,
                node_size=node_size_map_2d[agent_type],
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

        type_legend = [
            Line2D([0], [0], marker=m, color="w", label=t,
                markerfacecolor="gray", markeredgecolor="black", markersize=6)
            for t, m in type_marker_map.items()
        ]
        state_legend = [
            Line2D([0], [0], marker="o", color="w", label="Not adopted",
                markerfacecolor="#e74c3c", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Considering",
                markerfacecolor="#f39c12", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Adopted",
                markerfacecolor="#27ae60", markeredgecolor="black", markersize=8),
        ]

        leg1 = ax.legend(handles=type_legend, loc="upper left", bbox_to_anchor=(1.02, 1), title="Agent Type", borderaxespad=0, frameon=False)
        leg1.get_title().set_fontweight("bold")
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=state_legend, loc="upper left", bbox_to_anchor=(1.02, 0.65), title="State", borderaxespad=0, frameon=False)
        leg2.get_title().set_fontweight("bold")

   animation = FuncAnimation(
        fig,
        draw_frame,
        frames=list(model_data.index),
        interval=600,
        repeat=False
    )
   components.html(
    animation.to_jshtml(),
    height=1000,
    )


#### 3D visualization of the organizational network ####
st.subheader("3D Organizational Network")
selected_step = st.slider(
    "Select simulation step",
    0,
    int(model_data.index.max()),
    int(model_data.index.max())
)

frame_data = agent_data.loc[selected_step]
pos3d = {}

n = model.N

for i, node in enumerate(model.network.nodes()):

    phi = np.arccos(1 - 2*(i + 0.5)/n)
    theta = np.pi * (1 + np.sqrt(5)) * (i + 0.5)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    pos3d[node] = (x, y, z)
    u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 30)

x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

sphere_trace = go.Surface(
    x=x_sphere,
    y=y_sphere,
    z=z_sphere,
    opacity=0.10,
    showscale=False,
    hoverinfo="skip",
    showlegend=False
)
edge_x = []
edge_y = []
edge_z = []

for u, v in model.network.edges():

    x0, y0, z0 = pos3d[u]
    x1, y1, z1 = pos3d[v]

    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    line=dict(
        width=1,
        color="gray"
    ),
    hoverinfo="none",
    showlegend=False
)
color_map = {
    0: "#e74c3c",
    1: "#f39c12",
    2: "#27ae60"
}

size_map = {
    "Policy Maker": 16,
    "Manager": 18,
    "Receptive User": 12,
    "Neutral User": 12
}

type_symbol_map_3d = {
    "Policy Maker": "cross",
    "Receptive User": "circle",
    "Neutral User": "square",
    "Manager": "diamond"
}

nodes_by_type = {t: {"x": [], "y": [], "z": [], "colors": [], "sizes": [], "hover": [], "labels": []} for t in type_symbol_map_3d}

for node_id, agent in model.node_to_agent.items():
    x, y, z = pos3d[node_id]
    state = frame_data.loc[agent.unique_id]["State"]
    t = agent.agent_type
    nodes_by_type[t]["x"].append(x)
    nodes_by_type[t]["y"].append(y)
    nodes_by_type[t]["z"].append(z)
    nodes_by_type[t]["colors"].append(color_map[state])
    nodes_by_type[t]["sizes"].append(size_map[t])
    nodes_by_type[t]["labels"].append(str(node_id))
    nodes_by_type[t]["hover"].append(
        f"Agent {node_id}<br>Type: {agent.agent_type}<br>State: {state}<br>"
        f"Openness: {agent.openness:.2f}<br>Resistance: {agent.resistance:.2f}<br>Influence: {agent.influence:.2f}"
    )

node_traces = [
    go.Scatter3d(
        x=nodes_by_type[t]["x"],
        y=nodes_by_type[t]["y"],
        z=nodes_by_type[t]["z"],
        mode="markers+text",
        name=t,
        text=nodes_by_type[t]["labels"],
        textposition="top center",
        textfont=dict(size=9, color="black"),
        hovertext=nodes_by_type[t]["hover"],
        hoverinfo="text",
        showlegend=False,
        marker=dict(
            symbol=type_symbol_map_3d[t],
            size=nodes_by_type[t]["sizes"],
            color=nodes_by_type[t]["colors"],
            opacity=0.9
        )
    )
    for t in type_symbol_map_3d
]

type_legend_traces = [
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        name=t,
        legend="legend",
        marker=dict(size=10, color="gray", symbol=type_symbol_map_3d[t]),
        showlegend=True
    )
    for t in type_symbol_map_3d
]

state_legend_traces = [
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        name=label,
        legend="legend2",
        marker=dict(size=10, color=color, symbol="circle"),
        showlegend=True
    )
    for label, color in [("Not adopted", "#e74c3c"), ("Considering", "#f39c12"), ("Adopted", "#27ae60")]
]

fig3d = go.Figure(
    data=[sphere_trace, edge_trace] + node_traces + type_legend_traces + state_legend_traces
)

fig3d.update_layout(
    height=850,
    margin=dict(l=0, r=0, t=30, b=0),
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    legend=dict(
        title=dict(text="<b>Agent Type</b>"),
        x=1.0,
        y=0.7
    ),
    legend2=dict(
        title=dict(text="<b>State</b>"),
        x=1.0,
        y=0.5
    )
)

st.markdown('<style>div[data-testid="stPlotlyChart"] { margin-top: -8rem; }</style>', unsafe_allow_html=True)
st.plotly_chart(
    fig3d,
    use_container_width=True
)