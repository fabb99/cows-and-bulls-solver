import pytest

from src.analysis.best_model import pick_best_model
from src.analysis.tree_io import save_tree_to_json


def _tree(depth):
    # catena: ogni livello ha una foglia (vittoria) e un figlio, finché depth = 1
    node = {"guess": "0123", "possible_solutions": ["0123"], "children": {(4, 0): None}}
    for _ in range(depth - 1):
        node = {"guess": "0123", "possible_solutions": ["0123"], "children": {(4, 0): None, (0, 1): node}}
    return node


def test_pick_best_model_by_total_moves_then_name(tmp_path):
    save_tree_to_json(_tree(3), str(tmp_path / "estimated_evm_selected.json"))
    save_tree_to_json(_tree(2), str(tmp_path / "estimated_evm_geometric.json"))
    save_tree_to_json(_tree(2), str(tmp_path / "estimated_evm_piecewise.json"))
    kind, stats = pick_best_model(tmp_path)
    assert kind == "geometric" and stats["worst"] == 2
    with pytest.raises(FileNotFoundError):
        pick_best_model(tmp_path / "vuota")