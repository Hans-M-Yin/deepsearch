from collections import Counter

from debug.list_isolated_nodes import (
    _average_degrees,
    _node_degrees,
)


def test_node_degrees_counts_incoming_outgoing_and_self_loop() -> None:
    edges = [
        {"src_node_id": "a", "dst_node_id": "b"},
        {"src_node_id": "b", "dst_node_id": "b"},
    ]

    assert _node_degrees(edges) == Counter({"b": 3, "a": 1})


def test_average_degrees_by_requested_category() -> None:
    nodes = [
        {"node_id": "wiki-1", "node_type": "image", "source": {"source_type": "wikipedia_inline_image"}},
        {"node_id": "wiki-2", "node_type": "image", "metadata": {"image_origin": "wikipedia_inline"}},
        {"node_id": "visual-1", "node_type": "image", "source": {"source_type": "image_search"}},
        {"node_id": "text-1", "node_type": "text"},
        {"node_id": "text-2", "node_type": "text"},
        {"node_id": "other", "node_type": "image"},
    ]
    degrees = Counter({"wiki-1": 2, "wiki-2": 4, "visual-1": 5, "text-1": 1})

    assert _average_degrees(nodes, degrees) == {
        "wiki_inline": {"node_count": 2, "average_degree": 3.0},
        "visual_plan": {"node_count": 1, "average_degree": 5.0},
        "text": {"node_count": 2, "average_degree": 0.5},
    }
