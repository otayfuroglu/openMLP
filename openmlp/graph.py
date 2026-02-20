from langgraph.graph import END, StateGraph

from openmlp.non_equilibrium import non_equilibrium_node
from openmlp.qm import qm_calculation_node
from openmlp.state import PipelineState
from openmlp.train_nequip import train_nequip_node


def build_step1_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate_non_equilibrium", non_equilibrium_node)
    graph.set_entry_point("generate_non_equilibrium")
    graph.add_edge("generate_non_equilibrium", END)
    return graph.compile()


def build_step2_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate_non_equilibrium", non_equilibrium_node)
    graph.add_node("run_qm", qm_calculation_node)
    graph.set_entry_point("generate_non_equilibrium")
    graph.add_edge("generate_non_equilibrium", "run_qm")
    graph.add_edge("run_qm", END)
    return graph.compile()


def build_step3_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate_non_equilibrium", non_equilibrium_node)
    graph.add_node("run_qm", qm_calculation_node)
    graph.add_node("train_nequip", train_nequip_node)
    graph.set_entry_point("generate_non_equilibrium")
    graph.add_edge("generate_non_equilibrium", "run_qm")
    graph.add_edge("run_qm", "train_nequip")
    graph.add_edge("train_nequip", END)
    return graph.compile()
