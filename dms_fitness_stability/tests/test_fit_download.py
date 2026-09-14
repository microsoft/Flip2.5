import pytest

from dms_fitness_stability.fit_download import (
    build_download_actions,
    ensure_capacity,
    parse_human_size,
    required_bytes,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "family": "large_clean",
            "job_id": 63,
            "job_name": "large_clean_63",
            "issues": [],
            "artifacts": [
                {
                    "kind": "sampled_mle",
                    "experiment": "k50-mle",
                    "job_name": "k50_mle_lr0-01",
                    "name": "k50/k50_None_None_None_samples.nc",
                    "size": "443.6 GB",
                }
            ],
        }
    ]


def test_parse_human_size_uses_decimal_amulet_units():
    assert parse_human_size("555.5 MB") == 555_500_000
    assert parse_human_size("443.6 GB") == 443_600_000_000


def test_build_download_actions_uses_exact_remote_path_and_owner(tmp_path):
    actions = build_download_actions(
        _records(),
        amlt_executable="/opt/amlt",
        output_root=tmp_path,
        n_workers=1,
    )

    assert len(actions) == 1
    action = actions[0]
    assert action.destination == (
        tmp_path
        / "k50-mle/k50_mle_lr0-01/k50/k50_None_None_None_samples.nc"
    )
    assert action.command == (
        "/opt/amlt",
        "results",
        "download",
        "k50-mle",
        ":k50_mle_lr0-01",
        "--include",
        "k50/k50_None_None_None_samples.nc",
        "--output",
        str(tmp_path),
        "--n-workers",
        "1",
    )
    assert required_bytes(actions) == 443_600_000_000


def test_ensure_capacity_preserves_reserve(tmp_path):
    actions = build_download_actions(
        _records(),
        amlt_executable="/opt/amlt",
        output_root=tmp_path,
        n_workers=1,
    )

    with pytest.raises(OSError, match="require about 443.6 GB"):
        ensure_capacity(
            actions,
            free_bytes=500_000_000_000,
            reserve_bytes=100_000_000_000,
        )
