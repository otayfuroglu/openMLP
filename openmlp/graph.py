from langgraph.graph import END, StateGraph

from openmlp.non_equilibrium import non_equilibrium_node
from openmlp.qm import qm_calculation_node
from openmlp.state import PipelineState


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
