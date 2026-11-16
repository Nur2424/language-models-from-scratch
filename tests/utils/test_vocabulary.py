import pytest
import torch

from utils.vocabulary import BOUNDARY_ID, IGNORE_INDEX, Vocabulary, VocabularyError


def test_deterministic_ids_independent_of_order():
    a = Vocabulary.from_names(["cab", "bad"])
    b = Vocabulary.from_names(["dab", "abc", "ab"])
    assert a.chars == b.chars == ("a", "b", "c", "d")
    assert a.stoi == {".": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    assert a.itos[0] == "." and a.itos[4] == "d"


def test_token_conventions(names):
    v = Vocabulary.from_names(names)
    assert v.boundary_id == v.bos_id == v.eos_id == BOUNDARY_ID == 0
    assert v.num_classes == len(v.chars) + 1
    assert v.pad_id == v.num_classes            # input-only, outside the output space
    assert v.num_input_ids == v.num_classes + 1
    assert IGNORE_INDEX == -100 and IGNORE_INDEX not in v.itos
    assert v.pad_id not in v.itos


def test_encode_decode_round_trip(names):
    v = Vocabulary.from_names(names)
    for n in names:
        assert v.decode(v.encode(n)) == n
        framed = v.encode(n, add_bos=True, add_eos=True)
        assert framed[0] == framed[-1] == 0 and len(framed) == len(n) + 2
        assert v.decode(framed) == n
        assert v.decode(torch.tensor(framed)) == n


def test_decode_stops_at_eos_and_raw_mode():
    v = Vocabulary.from_names(["ab"])
    assert v.decode([0, 1, 2, 0, 1]) == "ab"
    assert v.decode([0, 1, 0, 2], strip_bos=False, stop_at_eos=False) == ".a.b"


def test_errors():
    v = Vocabulary.from_names(["ab"])
    with pytest.raises(VocabularyError, match="unknown character 'z'"):
        v.encode("az")
    with pytest.raises(VocabularyError, match="boundary"):
        v.encode("a.b")
    with pytest.raises(VocabularyError, match="PAD"):
        v.decode([1, v.pad_id])
    with pytest.raises(VocabularyError, match="outside"):
        v.decode([1, 99])
    with pytest.raises(VocabularyError, match="boundary token"):
        Vocabulary.from_names(["a.b"])
    with pytest.raises(VocabularyError):
        Vocabulary.from_names([])
    with pytest.raises(VocabularyError, match="sorted"):
        Vocabulary(chars=("b", "a"))


def test_serialisation_round_trip(names):
    v = Vocabulary.from_names(names)
    assert Vocabulary.from_dict(v.to_dict()) == v
