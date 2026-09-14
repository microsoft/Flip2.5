import h5py
import numpy as np
import pandas as pd

from dms_fitness_stability import (
    AnalysisIdentity,
    ScoreEnsemble,
    build_hmc_score_ensemble,
)
from dms_fitness_stability.hmc_scanning import sample_ids
from dms_fitness_stability.hmc_score_extraction import (
    audit_hmc_score_source,
    project_score_bundle_bytes,
)


def test_build_hmc_score_ensemble_matches_dense_chain_major_oracle(tmp_path):
    hmc_path = tmp_path / "fit.nc"
    native = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
    native[1, 1, 3] = np.inf
    with h5py.File(hmc_path, "w") as handle:
        posterior = handle.create_group("posterior")
        posterior.create_dataset(
            "log_r_mean",
            data=native,
            chunks=(1, 2, 2),
        )

    variants = pd.DataFrame(
        {
            "variant_id": ["A", "B", "C", "D", "E"],
            "hmc_variant_index": np.arange(5, dtype=np.int32),
            "reported_fitness": np.arange(5, dtype=float),
        }
    )
    analysis_variants = variants.iloc[[0, 2, 4]].copy().reset_index(drop=True)
    analysis_variants.insert(
        0,
        "matrix_variant_index",
        np.array([0, 2, 4], dtype=np.int64),
    )
    identity = AnalysisIdentity(
        protein_family="pdz",
        landscape="pdz-cis",
        inference_method="hmc",
        model_spec="exponential-exponential",
        fit_id="fit",
    )

    ensemble = build_hmc_score_ensemble(
        hmc_path=hmc_path,
        identity=identity,
        posterior_variable="log_r_mean",
        posterior_transform="identity",
        score_name="log_mean_growth",
        score_contract="test-identity-v1",
        variants=variants,
        analysis_variants=analysis_variants,
        transform=lambda values: values,
        n_samples=4,
        seed=23,
        sample_view="release_4",
        sample_chunk_size=2,
        variant_chunk_size=2,
        scratch_dir=tmp_path / "scratch",
        show_progress=False,
    )
    temporary_path = ensemble.temporary_paths[0]
    try:
        selected = sample_ids(6, 4, seed=23)
        expected = native.reshape(6, 5)[selected].astype(np.float32)
        expected[~np.isfinite(expected)] = np.nan
        np.testing.assert_equal(ensemble.fitness_score.values[0], expected)
        np.testing.assert_array_equal(
            ensemble.scores["source_sample_id"],
            selected,
        )
        np.testing.assert_array_equal(
            ensemble.scores["source_chain"] * 3 + ensemble.scores["source_draw"],
            selected,
        )
        assert temporary_path.is_file()

        output_dir = ensemble.save(tmp_path / "bundle")
        loaded = ScoreEnsemble.load(output_dir)
        try:
            np.testing.assert_equal(loaded.fitness_score.values[0], expected)
            audit = audit_hmc_score_source(
                loaded,
                hmc_path,
                sample_chunk_size=2,
                variant_chunk_size=2,
            )
            assert audit["status"] == "exact"
            assert audit["source_sha256"] == audit["matrix_sha256"]
        finally:
            loaded.close()
    finally:
        ensemble.close()

    assert not temporary_path.exists()


def test_project_score_bundle_bytes_scales_measured_scores_with_safety_margin(
    tmp_path,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "metadata.json").write_text('{"n_samples": 1000}', encoding="utf-8")
    (bundle / "fitness_scores.nc").write_bytes(b"x" * 1_000)
    (bundle / "variants.csv").write_bytes(b"y" * 100)

    assert project_score_bundle_bytes(
        bundle,
        target_samples=2_000,
        safety_factor=1.2,
    ) == 2_543
