from debug.list_isolated_nodes import _connected_node_ids


def test_connected_node_ids_collects_both_edge_endpoints() -> None:
    edges = [
        {"src_node_id": "a", "dst_node_id": "b"},
        {"src_node_id": "b", "dst_node_id": "c"},
        {"src_node_id": None, "dst_node_id": "d"},
    ]

    assert _connected_node_ids(edges) == {"a", "b", "c", "d"}


def test_connected_node_ids_ignores_missing_and_empty_endpoints() -> None:
    edges = [{}, {"src_node_id": "", "dst_node_id": None}]

    assert _connected_node_ids(edges) == set()
