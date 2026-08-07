from __future__ import annotations

from daily_research.path_policy import seq100_hot_path_pairs as pairs


def test_pair_contract_requires_activation_and_quality_coordinates() -> None:
    study = pairs.load_study()
    search = study["pair_search"]
    assert search["activation_features"]
    assert search["quality_features"]
    assert min(search["allowed_activation_bins"]) >= 2
    assert study["analysis"]["profit_claim_allowed"] is False


def test_pair_bin_round_trip() -> None:
    encoded = pairs._encode_pair_bin(3, 1, quintile_count=5)
    assert encoded == 16
    assert pairs._decode_pair_bin(encoded, quintile_count=5) == (3, 1)
