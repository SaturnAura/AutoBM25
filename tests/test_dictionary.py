import json
from pathlib import Path

from autobm25 import ParamDictionary
from autobm25.dictionary import FEATURE_KEYS


ROOT = Path(__file__).resolve().parents[1]
DICTIONARY_PATH = ROOT / "autobm25" / "data" / "param_dictionary.json"


def test_shipped_dictionary_shape():
    data = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))

    assert len(data["entries"]) == 48
    assert len(FEATURE_KEYS) == 21
    assert data["feature_keys"] == FEATURE_KEYS
    assert data["k_neighbors"] == 1


def test_every_dictionary_entry_has_valid_parameters():
    dictionary = ParamDictionary.load()

    for entry in dictionary.entries:
        params = entry["params"]
        assert {"k1", "b", "k3", "delta", "idf_type"} <= params.keys()
        assert 0 <= params["k1"] <= 3
        assert 0 <= params["b"] <= 1
        assert 0 <= params["k3"] <= 8
        assert 0 <= params["delta"] <= 2
        assert params["idf_type"] in {"rsj", "smoothed"}


def test_each_entry_is_its_own_nearest_neighbor():
    dictionary = ParamDictionary.load()

    for entry in dictionary.entries:
        probe = dictionary.probe(entry["features"])
        assert probe["distance"] == 0.0
