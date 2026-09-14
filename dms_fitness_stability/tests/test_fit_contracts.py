from pathlib import Path
from types import SimpleNamespace

from dms_fitness_stability.fit_contracts import (
    FitContractResolver,
    score_variable_for_job,
)
from dms_fitness_stability.fit_inventory import SelectedJob


def _job(dataset: str, subset: str, prior: str) -> SelectedJob:
    return SelectedJob(
        job_id=1,
        family="small_clean",
        dataset=dataset,
        subset=subset,
        fitness_distribution=prior,
        growth_curve="exponential",
        result="Perfect!",
        require_hmc=True,
        require_mle=True,
    )


def test_score_variable_follows_dataset_model_contracts():
    assert score_variable_for_job(_job("pdz", "cis", "lomax")) == "log_r_mean"
    assert score_variable_for_job(_job("pdz", "cript-c", "gamma")) == "inv_r_mean"
    assert score_variable_for_job(_job("trpb", "libA", "lomax")) == "r"
    assert score_variable_for_job(_job("trpb", "libD", "lomax")) == "log_r_mean"
    assert score_variable_for_job(_job("trpb", "libD", "gamma")) == "inv_r_mean"
    assert score_variable_for_job(_job("trpb", "four-site", "gamma")) == "inv_r_mean"
    assert score_variable_for_job(_job("nuc", "G1", "exponential")) == (
        "absolute_mean_log_fluorescence"
    )
    assert score_variable_for_job(_job("nuc", "G4", "lomax")) == "log_fluorescence"
    assert score_variable_for_job(_job("k50", "", "")) == "mean_dg"


def test_sampled_mle_contract_uses_one_chain_and_1000_draws(tmp_path):
    resolver = FitContractResolver(
        repository_root=Path(__file__).resolve().parents[1],
        data_root=tmp_path,
    )
    resolver._landscapes[("nuc", "G4")] = SimpleNamespace(
        n_fitted_variants=4,
        counts={},
    )
    nuc_expectation = resolver.expectation(
        _job("nuc", "G4", "lomax"),
        method="mle",
    )
    k50_job = SelectedJob(
        job_id=63,
        family="large_clean",
        dataset="k50",
        subset="",
        fitness_distribution="",
        growth_curve="",
        result="",
        require_hmc=False,
        require_mle=True,
    )
    k50_expectation = resolver.expectation(k50_job, method="mle")

    assert nuc_expectation.posterior_shape == (1, 1_000, 4)
    assert k50_expectation.posterior_shape == (1, 1_000, 1_824_523)


def test_four_site_mle_contract_uses_fitted_variant_count(tmp_path):
    count_path = tmp_path / "trpb/raw/counts/four-site.csv"
    count_path.parent.mkdir(parents=True)
    lines = [
        "Time (h),AAs,InputCount_1,InputCount_2,OutputCount_1,OutputCount_2"
    ]
    variants = ("AAAA", "AAAC", "AAAD")
    for time in (12, 16, 20, 24, 28, 36):
        for index, variant in enumerate(variants, start=1):
            lines.append(
                f"{time},{variant},{index},{index + 1},{index + 2},{index + 3}"
            )
    count_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reported_path = tmp_path / "trpb/raw/reported_fitness/four-site.csv"
    reported_path.parent.mkdir(parents=True)
    reported_path.write_text(
        "AAs,fitness\nAAAA,1.0\nAAAC,0.5\nAAAD,0.0\n",
        encoding="utf-8",
    )
    resolver = FitContractResolver(
        repository_root=Path(__file__).resolve().parents[1],
        data_root=tmp_path,
    )
    job = SelectedJob(
        job_id=64,
        family="large_clean",
        dataset="trpb",
        subset="four-site",
        fitness_distribution="exponential",
        growth_curve="exponential",
        result="",
        require_hmc=False,
        require_mle=True,
    )

    expectation = resolver.expectation(job, method="mle")

    assert expectation.score_variable == "log_r_mean"
    assert expectation.variant_count == 3
    assert expectation.posterior_shape == (1, 1_000, 3)
    assert expectation.predictive_variables == ()
    assert expectation.observed_arrays == ()
