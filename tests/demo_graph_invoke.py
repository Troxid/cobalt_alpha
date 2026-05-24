from cobalt_alpha.graph import GraphState, build_graph


def test_graph_1():
    state = GraphState(raw_user_input="привет")
    graph = build_graph()
    resp = graph.invoke(state)
    print(resp)


def test_graph_2():
    state = GraphState(raw_user_input="что такое закон Ома")
    graph = build_graph()
    resp = graph.invoke(state)
    print(resp)
