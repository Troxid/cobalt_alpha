from cobalt_alpha.graph import build_graph
from cobalt_alpha.state import GraphState


def test_graph_1():
    state = GraphState(current_user_input="привет")
    graph = build_graph()
    resp = graph.invoke(state)
    print(resp)


def test_graph_2():
    state = GraphState(current_user_input="что такое закон Ома")
    graph = build_graph()
    resp = graph.invoke(state)
    print(resp)
