# Exemplars, verbatim, from the group's papers

Quoted from arXiv versions of: SnapMMD (2505.16082), Helmholtz GP (2302.10364, ICML 2023), SBIRR (2408.06277, AISTATS 2025), Smooth Sailing (2502.06067, NeurIPS 2025), PM2.5 forecasts (2409.05866, BAMS 2026), AMIP (2011.14999), Sensitivity of MCMC (2408.07240), Oaxaca–Blinder reference choice (2603.29972), LLM-ranking sensitivity (2508.11847, ICLR 2026). Use these as side-by-side comparisons when drafting; do not copy sentences into a new paper.

Contents: 1 Abstracts · 2 Intro openers · 3 Naming the baseline and the gap · 4 Contributions paragraphs · 5 Related-work prose · 6 Setup / notation · 7 Theory framing · 8 Experiment openers · 9 Figure references and captions · 10 Discussion and limitations · 11 Practical questions with a named actor

---

## 1. Abstracts

**SnapMMD** (methods paper, ML venue):

> Scientists often want to make predictions beyond the observed time horizon of "snapshot" data following latent stochastic dynamics. For example, in time course single-cell mRNA profiling, scientists have access to cellular transcriptional state measurements (snapshots) from different biological replicates at different time points, but they cannot access the trajectory of any one cell because measurement destroys the cell. Researchers want to forecast (e.g.) differentiation outcomes from early state measurements of stem cells. Recent Schrödinger-bridge (SB) methods are natural for interpolating between snapshots. But past SB papers have not addressed forecasting — likely since existing methods either (1) reduce to following pre-set reference dynamics (chosen before seeing data) or (2) require the user to choose a fixed, state-independent volatility since they minimize a Kullback-Leibler divergence. Either case can lead to poor forecasting quality. In the present work, we propose a new framework, SnapMMD, that learns dynamics by directly fitting the joint distribution of both state measurements and observation time with a maximum mean discrepancy (MMD) loss. Unlike past work, our method allows us to infer unknown and state-dependent volatilities from the observed data. We show in a variety of real and synthetic experiments that our method delivers accurate forecasts. Moreover, our approach allows us to learn in the presence of incomplete state measurements and yields an $R^2$-style statistic that diagnoses fit. We also find that our method's performance at interpolation (and general velocity-field reconstruction) is at least as good as (and often better than) state-of-the-art in almost all of our experiments.

**Helmholtz GP** (methods paper with theory):

> Given sparse observations of buoy velocities, oceanographers are interested in reconstructing ocean currents away from the buoys and identifying divergences in a current vector field. As a first and modular step, we focus on the time-stationary case - for instance, by restricting to short time periods. Since we expect current velocity to be a continuous but highly non-linear function of spatial location, Gaussian processes (GPs) offer an attractive model. But we show that applying a GP with a standard stationary kernel directly to buoy data can struggle at both current reconstruction and divergence identification, due to some physically unrealistic prior assumptions. To better reflect known physical properties of currents, we propose to instead put a standard stationary kernel on the divergence and curl-free components of a vector field obtained through a Helmholtz decomposition. We show that, because this decomposition relates to the original vector field just via mixed partial derivatives, we can still perform inference given the original data with only a small constant multiple of additional computational expense. We illustrate the benefits of our method with theory and experiments on synthetic and real ocean data.

**SBIRR** (note the sentence-initial But / So / And):

> Practitioners often aim to infer an unobserved population trajectory using sample snapshots at multiple time points. E.g., given single-cell sequencing data, scientists would like to learn how gene expression changes over a cell's life cycle. But sequencing any cell destroys that cell. So we can access data for any particular cell only at a single time point, but we have data across many cells. The deep learning community has recently explored using Schrödinger bridges (SBs) and their extensions in similar settings. However, existing methods either (1) interpolate between just two time points or (2) require a single fixed reference dynamic (often set to Brownian motion within SBs). But learning piecewise from adjacent time points can fail to capture long-term dependencies. And practitioners are typically able to specify a model family for the reference dynamic but not the exact values of the parameters within it. So we propose a new method that (1) learns the unobserved trajectories from sample snapshots across multiple time points and (2) requires specification only of a family of reference dynamics, not a single fixed one. We demonstrate the advantages of our method on simulated and real data.

**Smooth Sailing** (theory-forward statistics paper):

> Linear models remain ubiquitous in modern spatial applications — including climate science, public health, and economics — due to their interpretability, speed, and reproducibility. While practitioners generally report a form of uncertainty, popular spatial uncertainty quantification methods do not jointly handle model misspecification and distribution shift — despite both being essentially always present in spatial problems. In the present paper, we show that existing methods for constructing confidence (or credible) intervals in spatial linear models fail to provide correct coverage due to unaccounted-for bias. In contrast to classical methods that rely on an i.i.d. assumption that is inappropriate in spatial problems, in the present work we instead make a spatial smoothness (Lipschitz) assumption. We are then able to propose a new confidence-interval construction that accounts for bias in the estimation procedure. We demonstrate that our new method achieves nominal coverage via both theory and experiments. Code to reproduce experiments is available at https://github.com/DavidRBurt/Lipschitz-Driven-Inference.

**PM2.5 forecasts** (applied / evaluation paper, domain journal):

> Wildfire frequency is increasing as the climate changes, and the resulting air pollution poses health risks. Just as people routinely use hourly weather forecasts to plan their day's activities around precipitation, reliable hourly air quality forecasts could help individuals reduce their exposure to air pollution. In the present work, we evaluate six existing forecasts of ground-level fine particulate matter (PM2.5) within the continental United States during the 2023 fire season. We include forecasts using physical simulation, ensembling, and artificial intelligence. We focus our evaluation on individual decisions, such as (1) whether to go outside on a day with potentially high PM2.5 or (2) when to go outside for the lowest PM2.5 exposure. Our evaluation consists of both visualizations of hourly PM2.5 forecasts in particular locations as well as metrics summarizing forecast skill for the two tasks above. As part of our analysis, we introduce a new evaluation metric for the task of deciding when to go outside. We find meaningful room for improvement in PM2.5 forecasting, which might be realized by improving physical models, incorporating more data sources, and using artificial intelligence tools.

**Oaxaca–Blinder reference choice** (Broderick; gap as an absence, negative-ish verdict stated flatly, recommendation at the end):

> Scientists often want to explain why an outcome is different in two groups. For instance, differences in patient mortality rates across two hospitals could be due to differences in the patients themselves (covariates) or differences in medical care (outcomes given covariates). The Oaxaca–Blinder decomposition (OBD) is a standard tool to tease apart these factors. It is well known that the OBD requires choosing one of the groups as a reference, and the numerical answer can vary with the reference. To the best of our knowledge, there has been no systematic investigation into whether the choice of OBD reference can yield different substantive conclusions and how common this issue is. In the present paper, we give existence proofs in real and simulated data that the OBD references can in fact yield substantively different conclusions. Our empirical exercises find that this sensitivity is more common when the OBD is extended to more complex regression models, including a pretrained transformer. Our theoretical and empirical results together establish that these conclusion reversals are not entirely driven by model misspecification, small data, or adversarial parameter choices. Our results suggest that practitioners should always report both directions of the OBD; that modern machine learning and large datasets do not automatically resolve the conclusion reversal problem; and that further work on this problem is needed.

**Sensitivity of MCMC** (Broderick; rhetorical question → naive idea → why infeasible; mixed result stated in the abstract):

> If the conclusion of a data analysis is sensitive to dropping very few data points, that conclusion might hinge on the particular data at hand rather than representing a more broadly applicable truth. How could we check whether this sensitivity holds? One idea is to consider every small subset of data, drop it from the dataset, and re-run our analysis. But running MCMC to approximate a Bayesian posterior is already very expensive; running multiple times is prohibitive, and the number of reruns needed here is combinatorially large. Recent work proposes a fast and accurate approximation to find the worst-case dropped data subset, but that work was developed for problems based on estimating equations and does not directly handle Bayesian posterior approximations using MCMC. We make two principal contributions in the present work. We adapt the existing data-dropping approximation to estimators computed via MCMC. Observing that Monte Carlo errors induce variability in the approximation, we use a variant of the bootstrap to quantify this uncertainty. We demonstrate how to use our approximation in practice to determine whether there is non-robustness in a problem. Empirically, our method is accurate in simple models, such as linear regression. In models with complicated structure, such as hierarchical models, the performance of our method is mixed.

**LLM-ranking sensitivity** (ICLR; venue-typical "We propose" opener, headline number as count and percentage):

> We propose a method for evaluating the robustness of widely used LLM ranking systems -- variants of a Bradley--Terry model -- to dropping a worst-case very small fraction of preference data. Our approach is computationally fast and easy to adopt. When we apply our method to matchups from popular LLM ranking platforms, including Chatbot Arena and derivatives, we find that the rankings of top-performing models can be remarkably sensitive to the removal of a small fraction of preferences; for instance, dropping just 0.003% of human preferences can change the top-ranked model on Chatbot Arena. Our robustness check identifies the specific preferences most responsible for such ranking flips, allowing for inspection of these influential preferences.

---

## 2. Introduction openers

**SnapMMD** (define terms inline in prose; two "for instance"s):

> Many scientific modeling problems require forecasting stochastic dynamics from snapshot data. Here, snapshot data represents observations taken at different time points, without access to individual trajectories. And forecasting represents predicting future states beyond the observed times. For example, single-cell RNA sequencing (scRNA-seq) is widely used to study dynamic processes such as development, immune activation, and cancer progression. Each measurement in scRNA-seq destroys the cell, so scientists observe independent biological replicates at discrete times rather than a single replicate across multiple times. Despite the absence of individual cell trajectories, researchers often aim to forecast future cellular states; for instance, researchers are interested in forecasting differentiation outcomes of stem cells, immune cell activity after initial signal stimulation, or long-term cancer cell response to drugs with transcriptomic snapshots taken shortly after treatments. A common additional challenge is incomplete state measurement. For instance, although protein expression level mediates the dynamics of gene expression, protein level cannot be measured by scRNA-seq.

**Smooth Sailing** (three cited real examples, then the specific inferential goal):

> Scientists often want to understand the sign and magnitude of an association in cases where covariates vary spatially. Examples include the relationship between aerosol concentrations and regional precipitation changes (Westervelt et al., 2018), the association between proximity to major highways and the prevalence of dementia (Li et al., 2023), and the link between air pollution exposure and birth weight (Lee et al., 2022). Here, the covariates (aerosol concentrations, proximity to highways, and air pollution) may all be thought of as a function of spatial location. Moreover, scientists often have access to data at some spatial locations but want to infer the association of interest at other locations. For instance, a country might measure birth weight and air pollution at the municipal level for some municipalities — and be interested in understanding the association between air pollution and birth weight in unobserved municipalities.

**Oaxaca–Blinder** (named actor, running pronoun, closing question in the reader's voice):

> As a motivating example, suppose a data scientist at Hospital H notices that mortality rates are noticeably higher at Hospital H than at Hospital K. She knows the medical literature links many risk factors (X), such as high blood pressure, to mortality (Y). The observed differences in mortality could be due to differences in these risk factors between the two groups of patients (distribution of X) or to differences in how well each hospital manages these risk factors (distribution of Y∣X). If she can figure out how much each of these components is driving the difference in mortality rates, her hospital may be able to change its policy to improve patient care. How can she use the available information to form hypotheses about what drives the difference in mortality?

**Sensitivity of MCMC** (a real published study as the running example; analyst and policymaker as actors):

> Consider this motivating example. Angelucci et al. [2015] conducted a randomized controlled trial (RCT) in Mexico to study whether microcredit improves business profits. One might choose to analyze the data from this RCT using a simple (but non-conjugate) Bayesian model and Markov chain Monte Carlo (MCMC). Based on an MCMC estimate of the posterior mean effect of microcredit, an analyst might conclude that microcredit actually reduces profit. So, microcredit might be viewed as detrimental to the businesses in this RCT. Next, if a policymaker wants to advocate against microcredit deployment outside of Mexico, they need to know if microcredit remains detrimental beyond the data gathered in Angelucci et al. [2015]. More broadly, many researchers analyze data with Bayesian models and MCMC [Senf et al., 2020, Meager, 2022, Jones et al., 2021, Porter et al., 2022] and want to know if their conclusions generalize beyond their data.

---

## 3. Naming the baseline and stating the gap

**Helmholtz GP** (fair description, then a name, then "However … we show there remains substantial room for improvement"):

> Gonçalves et al. (2019) and Lodise et al. (2020) proposed modeling buoy velocities in the latitude and longitude directions according to independent Gaussian processes (GPs) with standard spatiotemporal kernels (e.g., squared exponential kernels). In our work, we focus on the spatial aspects of this task and assume the velocity field is stationary in time. Even under this simplification, an independent spatial GP prior on the velocities is a natural choice due to its ability to handle the sparsity of buoy observations on the ocean surface and its assumption that currents vary continuously but in a nonlinear fashion. We call such a model the *velocity GP*.
>
> However, in what follows, we show that there remains substantial room for improvement. In simulated cases where we have access to ground truth, we observe that the velocity GP approach can fail to complete vortices or fail to connect currents when buoys are observed sparsely. And while we show how to derive divergence estimates for the velocity GP, we also find that these estimates often fail to capture the true divergence when it is known in simulations or real data.

**SnapMMD** (packed citations, "for two reasons", enumerated):

> Recent work has addressed the problem of interpolating between snapshots through Schrödinger bridge (SB) methods (Pavon et al., 2021; De Bortoli et al., 2021; Vargas et al., 2021; Koshizuka and Sato, 2022; Wang et al., 2024) and their multi-marginal extensions (Shen et al., 2025; Zhang, 2024; Guan et al., 2024; Chen et al., 2024; Lavenant et al., 2024). These methods reconstruct likely trajectories between snapshots at consecutive times. SB methods interpolate via regularized couplings, often chosen to be entropy-regularized. In settings where the goal is to fill in missing timepoints between observed snapshots, these techniques perform well. However, SB methods have not yet been systematically evaluated at forecasting, and we expect them to struggle for two reasons.

**LLM rankings** (naive solution dismissed on computational grounds; "So we instead turn to"):

> Our question posed above motivates the need for a systematic way to assess the robustness of top rankings in BT-based evaluation systems to worst-case data dropping. However, a brute-force combinatorial search over all possible small subsets of data would be computationally infeasible for large-scale platforms like Chatbot Arena. So we instead turn to a recent line of works from statistics and theoretical computer science that design algorithms for assessing whether data analyses are robust to dropping a small, worst-case fraction of data points (Broderick et al., 2020; Kuschnig et al., 2021; Moitra & Rohatgi, 2023; Freund & Hopkins, 2023; Shiffman et al., 2023; Nguyen et al., 2024; Huang et al., 2025a; Rubinstein & Hopkins, 2025).

---

## 4. Contributions paragraphs (prose, inline enumeration, section numbers inline)

**SnapMMD:**

> In this work, we propose a new framework, SnapMMD, that shifts the modeling focus from interpolation to accurate, interpretable forecasting. Our approach begins with the observation that in typical trajectory-inference settings (Lavenant et al., 2024), each sample can be viewed as an i.i.d. draw from the *joint* distribution of a system's state (e.g., mRNA expression level) and the time of measurement. Then, we characterize a parametric family of stochastic differential equations (SDEs) and seek the member of this family whose implied joint distribution over state and time best matches the empirical joint distribution observed in the data. We perform this matching using maximum mean discrepancy (MMD), a kernel-based measure of distance between distributions that allows us to compare the model-predicted state-time distribution to the observed one without requiring access to individual trajectories or likelihoods. With a specific kernel choice, we show that this matching problem reduces to a distributional least-squares objective. Our formulation yields three key benefits: (1) accurate forecasts beyond the observed time horizon, (2) robust interpolation and model learning even with incomplete state measurements, and (3) interpretable model outputs, including an explicit velocity field and an $R^2$-like metric that quantifies model fit.

Closing paragraph of the same intro (evidence preview + code; no roadmap):

> We evaluate SnapMMD across a range of synthetic and real-world systems, including a synthetic gene regulatory network and real time-course single-cell RNA sequencing datasets. Our primary focus is on forecasting, where SnapMMD consistently outperforms Schrödinger bridge baselines, providing more accurate one-step ahead predictions beyond the observed time window. We also assess SnapMMD on interpolation tasks — inferring intermediate dynamics between observed snapshots. In these experiments, our method matches or exceeds the performance of both SB-based and flow-matching baselines in almost all cases. Crucially, SnapMMD handles incomplete state measurements with ease, reconstructs an interpretable velocity field that supports downstream scientific analysis, and provides an $R^2$-style metric to evaluate model fit. Code to reproduce the experiments is available at https://github.com/YunyiShen/snapMMD.

**Helmholtz GP** ("For one, … Second, … Finally, …"; "we prove", "we show", "we demonstrate"):

> The resulting *Helmholtz GP* prior offers several conceptual advantages. For one, oceanographers expect the two components to have substantially different magnitudes and length scales; it is straightforward to encode these differences with a Helmholtz GP. By contrast, we prove that the velocity GP implies an a priori belief that the divergent and rotational components have the same magnitude. Second, we expect correlation between the longitudinal and latitudinal components of a current, which the Helmholtz GP exhibits – and the velocity GP lacks by construction. Finally, the Helmholtz GP is agnostic to the (arbitrary) choice of reference frame, while the velocity GP is not necessarily. We demonstrate that the Helmholtz GP is amenable to practical inference. Since (i) the Helmholtz decomposition is based on partial (mixed) derivatives, and (ii) the derivative of a GP is a GP, we show that our prior choice implies a tractable GP prior on the current itself. Therefore, we can still perform inference given the original data with no extra approximation. And our method suffers no increase in computational complexity relative to the velocity GP. Finally, we demonstrate the superior performance of the Helmholtz GP at the current-reconstruction task and divergence-estimation task (as well as vorticity estimation) in a variety of experiments on simulated and real data. Code is available at https://github.com/renatoberlinghieri/Helmholtz-GP.

**Smooth Sailing** (a single "principal contribution" with (a)/(b)):

> Our principal contribution is to provide the first confidence intervals for spatial linear regression that are guaranteed to achieve frequentist coverage at the nominal level (a) under nonparametric assumptions about the data-generating distribution and (b) when the target locations need not match the observed data. Along the way, we formalize the problem of estimation in this setting. … In our experiments, our method is (by far) the only method to consistently achieve nominal coverage.

**Oaxaca–Blinder** (section numbers inline, replacing a roadmap):

> In the present work, after reviewing the OBD and the index number problem (Section 2), we give an existence proof (with real data) that the OBD *can* yield two substantively different conclusions (Section 3). We analyze a range of data (healthcare and U.S. Census data), and we show that the problem persists for the more complex, nonlinear extensions of the OBD. In fact, we show that these extensions, including one with a recent foundation model, seem to suffer much more frequently from the issue than a simple linear model. In Section 4, we further demonstrate theoretically that this phenomenon is not an artifact of model misspecification, small data size, or adversarial parameter choices. We suggest best practices for data analysts in Section 5.

**Sensitivity of MCMC** (run-in bold heading; honest caveat inside the contributions):

> **Our contributions.** We extend Broderick et al. [2023] to handle analyses based on MCMC. In section 2, we introduce necessary concepts in Bayesian decision-making, and we describe sensitivity to small-data dropping in more detail. In section 3.1, we form a first-order approximation to the effect of removing observations; to do so, we use known results on how much a posterior expectation locally changes under small perturbations to the total log likelihood [...]. Recognizing that Monte Carlo errors induce variability in our approximation, in section 3.3, we use a variant of the bootstrap Efron [1979] to quantify this uncertainty. … Experimentally, we apply our method to three Bayesian analyses. In section 6, we can detect non-robustness in econometric and ecological studies. However, while our approximation performs well in simple models such as linear regression, it is less reliable in complex models, such as ones with many random effects.

---

## 5. Related-work prose (each prior work followed by exactly what it does not do)

**Helmholtz GP**, run-in paragraph after the intro:

> **Related work.** The Helmholtz decomposition has been used for visualization and physical interpretation of an oceanographic field (Rocha et al., 2016; …). But these methods assume the velocity vector field is known on a grid whereas our goal is to reconstruct the vector field from sparse observations. Prior work in atmospheric statistics used a Helmholtz decomposition to perform regression on the residuals of a physical model (Daley, 1985; Hollingsworth & Lönnberg, 1986); this approach would give the same mean prediction as the Helmholtz GP if the same covariance function were used. However, to estimate the covariance function, the authors rely on a series representation and binning procedure; … Wahlström (2015) proposes independent GP priors on the terms in a Helmholtz decomposition. But the authors assume direct access to noisy observations of each of the divergence-free and curl-free components separately - whereas we aim to recover the individual components from noisy observations of their sum. …
>
> In sum, then, it is not clear from existing work that divergence and vorticity can be usefully or practically recovered when observations come from a general (noisy) vector field that is neither curl- nor divergence-free. … We discuss related work further in Appendix A.

**LLM rankings** (contrast as a numbered list):

> Our notion of non-robustness differs from those of Min et al. (2025); Huang et al. (2025b); Zhao et al. (2025) in two major respects. First, it occurs at a different place in the process, at the data analysis step after data has been collected (including from malicious or apathetic users). Second, it does not require adversarial intent.

**Sensitivity of MCMC**, Section 1.1 opener: "Our work arguably fits into the intersection of three lines of work. We have already mentioned the first: papers on detecting sensitivity to small-data removal."

---

## 6. Setup / notation

> Though our work has application beyond scRNA-seq, we next describe our data and goals using scRNA-seq terminology to clarify and concretize our notation. **Data Setup.** We consider single-cell mRNA measurements collected at I distinct time points, labeled $t_1 < t_2 < \cdots < t_I$. (SnapMMD)

> We consider a dataset D of M observations, $\{(\mathbf{x}_m,\mathbf{y}_m)\}_{m=1}^M$. Here $\mathbf{x}_m=(x_m^{(1)},x_m^{(2)})^{\top}\in\mathbb{R}^2$ represents the location of a buoy, typically a longitude and latitude pair. (Helmholtz GP)

> In addition to the time distribution h(t), we model each latent trajectory of the (i,n) cell with a stochastic differential equation (SDE) driven by a d-dimensional Brownian motion $W_t^{(i,n)}$, independent across particles: $dX_t^{(i,n)} = b_0(X_t^{(i,n)}, t)dt + g_0(X_t^{(i,n)}, t)dW_t$, $X_{t=0}^{(i,n)} \sim \pi_0$. (SnapMMD)

Run-in headings observed: Data Setup. / Goal. / Dataset and independence assumptions. / Model. / Our Estimand. / Review: Well-specified Linear Model. / Misspecified Spatial Setup: Data-Generating Process. / Metrics of success. / Forecasting (and vector field reconstruction) baselines. / Interpolation baselines. / Handling incomplete state measurements. / Choosing kernel, optimizer, and SDE model. / Our RKHS-based R² metric. / Lipschitz-Driven Confidence Intervals.

Method-section opener (contrast with the named baseline): "Instead of putting separate GP priors with isotropic kernels on the two components of F as in the velocity GP, we propose to put separate GP priors with isotropic kernels on the Helmholtz scalar potentials Φ and Ψ." (Helmholtz GP) / "To address the forecasting limitations of SBs described above, we propose an alternative approach. SBs compare the observed data and candidate model directly via Kullback–Leibler divergence. We instead formulate an optimization problem that directly matches the joint distribution of state–time pairs predicted by a candidate model to the empirical distribution observed in the data." (SnapMMD)

---

## 7. Theory framing

**SnapMMD, Proposition 3.1** — plain-English claim → statement → "(proof in Appendix B.2)" → interpretation → use:

> We choose a kernel that factors across state and time; this choice lets us break the joint MMD into a weighted sum of marginal MMDs at each time point. The resulting optimization objective is reminiscent of least-squares regression, with time acting as a discrete index.
>
> **Proposition 3.1.** Let $f(y,t) = f(y \mid t)h(t)$ and $g(y,t) = g(y \mid t)h(t)$ be joint distributions over $y \in \mathbb{R}^d$ (for dimension d) and discrete time $t \in \mathcal{T}$, where h(t) is a probability mass function and $f(y \mid t), g(y \mid t)$ are conditional distributions. Use the kernel $K((y,t),(y',t')) = K_y(y,y')\delta(t-t')$, where $K_y$ is positive definite on the state space and [moment conditions]. Then: $\mathrm{MMD}_{K}^{2}(f,g) = \sum_{t \in \mathcal{T}} h^{2}(t)\,\mathrm{MMD}_{K_{y}}^{2}(f(\cdot \mid t), g(\cdot \mid t)).$
>
> This result (proof in Appendix B.2) shows that aligning joint distributions here boils down to matching conditional state distributions across time. We apply Proposition 3.1 with the two joint distributions from above, $f_{\theta}(y,t)$ and $\hat{f}(y,t)$, to obtain the following optimization objective:

**Helmholtz GP, Propositions 4.1–4.2** — "In fact, … we show that"; counterexample flagged; interpretation then contrast with baseline:

> By contrast, the velocity GP provides no such control. In fact, for any isotropic choice of $k^{(1)}$ and $k^{(2)}$ we show that the resulting velocity GP must assume the same variance on the divergence and vorticity in the prior.
>
> **Proposition 4.1.** Let $k^{(1)}$ and $k^{(2)}$ be isotropic kernels with inputs $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^2$. Take $F^{(1)} \sim \mathcal{GP}(0, k^{(1)})$ and $F^{(2)} \sim \mathcal{GP}(0, k^{(2)})$ independent. Suppose $k^{(1)}$ and $k^{(2)}$ are such that $F^{(1)}$, $F^{(2)}$ have almost surely continuously differentiable sample paths. Let $\delta$ and $\zeta$ be defined as in Equations (3) and (4). Then for any $\mathbf{x}$, $\operatorname{Var}[\delta(\mathbf{x})] = \operatorname{Var}[\zeta(\mathbf{x})]$.
>
> The proof of Proposition 4.1 appears in Appendix H.1.

> For intuition in the other direction, consider the following counterexample. …
>
> Proposition 4.2 formalizes that it is equivalent to either (1) rotate the data and then predict using the Helmholtz GP or (2) predict using the Helmholtz GP and rotate the prediction. The equivariance property in Proposition 4.2 need not hold in general for velocity GP priors.

**Smooth Sailing** — numbered assumptions, each with a one-line lead-in; theorem re-lists them; "That is," gloss inside the statement:

> Under assumption 1, our data-generating process simplifies. **Assumption 1.** There exists a function $\chi : \mathcal{S} \to \mathbb{R}^P$ such that $X_m^{\star} = \chi(S_m^{\star})$ …
> Assumption 2 introduces the key simplification. **Assumption 2.** There exists a function $f: \mathcal{S} \to \mathbb{R}$ such that … where $\epsilon_m^{\star}, \epsilon_n \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$.
> To solve the minimization in Equation (6), it will be convenient to assume invertibility, as for OLS. **Assumption 3.** $X^{\star T}X^{\star}$ is invertible.
> Since we've seen that the classic assumptions are inappropriate in this spatial setting, we instead assume f is shared across source and target (assumption 2) and not varying so quickly in space as to be difficult to learn from limited data. **Assumption 4.** The conditional expectation, f, is L-Lipschitz …
>
> The resulting confidence interval appears in Algorithm 1. We next establish its validity. So far, we have covered only the known $\sigma^2$ case. We handle the unknown $\sigma^2$ case after the following theorem.
>
> **Theorem 7.** Suppose $(S_m^\star, X_m^\star, Y_m^\star)_{m=1}^M$ and $(S_n, X_n, Y_n)_{n=1}^N$ satisfy assumptions 1 to 4 with known $\sigma^2$. Define the (random) interval $I^{\Psi}$ as in Algorithm 1 using the known value of $\sigma^2$. Then with probability at least $1-\alpha$, $\theta^{\star}_{OLS,p} \in I^{\Psi}$. That is, $I^{\Psi}$ has coverage (conditional on the test locations) at least $1 - \alpha$.
>
> In Appendix E.2, we prove validity of our confidence interval for a generic choice of weight matrix $\Psi$. Theorem 7 is an immediate corollary of that result.

**SBIRR, Assumptions 1–2 and Proposition 1** — assumptions stated semi-formally with "i.e." glosses; proof pointer names the technique:

> **Assumption 1.** The drifts are L-Lipschitz; i.e., for all $t \in [0, t_I]$, $\|\boldsymbol{b}(x, t) - \boldsymbol{b}(y, t)\| \le L\|x - y\|$ … And we have at most linear growth; i.e., there exist $K < \infty$ and constant c such that $\|\boldsymbol{b}(y, t)\| < K\|y\| + c$.
> **Assumption 2.** At each time step $t_i$, the distribution of the $N_{t_i}$ particles has bounded second moments.
> … By construction, this iterative scheme will monotonically decrease the objective in eq. (6) at each step. **Proposition 1.** Suppose $\mathcal{F}$ is a convex set of densities … Take any initialization $\mathbf{b}_{ref}^{(0)}$. If … we have $\lim_{k \to \infty} D_{\mathrm{KL}}(\cdot) = \inf \cdots$.
> We prove this result in appendix C by adapting standard arguments from the theory of iterative projections (Csiszár, 1975; Csiszár and Shields, 2004; Benamou et al., 2015).

**Sensitivity of MCMC, Theorem 3.1** — "Take assumptions …"; interpretation names what each equation is:

> Under assumptions 2.1, 3.1 and 3.2, φ(w) is continuously differentiable with respect to w.
> **Theorem 3.1.** Take assumptions 2.1, 3.1 and 3.2. For any δ ∈ (0,1), φ(w) is continuously differentiable with respect to w on {w ∈ [0,1]^N : max_n w_n ≥ δ}. The n-th partial derivative at w equals c₁f + c₂h where …
> Theorem 3.1 establishes both the existence of the partial derivatives and their formula. Equation (2) is the partial derivative of the posterior mean with respect to the weights, while eq. (3) is that for the posterior standard deviation.

**Oaxaca–Blinder** — theory section opened by practitioner questions; named proposition; limitation next to the theory:

> However, a few major questions remain. First, do these conclusion reversals occur in the absence of finite-sample noise; i.e., do practitioners with large-scale datasets still need to worry about them? Second, even if conclusion reversals can occur, do they occur outside of mathematically adversarial examples (i.e., "measure zero" in some sense)?
> **Proposition 4.4** (OBD unexplained sign flips). The OBD unexplained component flips signs if and only if the following two conditions hold: …
> See Section D.1 for the proof. From Proposition 4.4, we see that group-specific slopes serve a different role in sign flips in the unexplained case relative to the explained case.
> One limitation of this theory is that it assumes regression parameters are uniformly distributed, when in practice we may expect they follow non-uniform distributions.

---

## 8. Experiment-section openers (verdict first)

> In synthetic and real-data experiments, we find that our SnapMMD method consistently provides better forecasts than competitors. We also find that, in almost all experiments, SnapMMD provides better or matching interpolation performance relative to competitors. In Appendix D.1, we further demonstrate that our method outperforms competitors on vector field reconstruction. (SnapMMD)

> We next empirically compare the SE-Helmholtz GP and SE-velocity GP. We find that the SE-Helmholtz GP yields better current predictions as well as better divergence (and vorticity) identification, across a variety of simulated and real data sets. We also compare to dissipative Hamiltonian neural networks (D-HNNs) (Greydanus & Sosanya, 2022) in Table 2 and Appendix I but find that the GP methods generally perform better. (Helmholtz GP)

> In simulated and real experiments, we find that our method gives more accurate predictions than alternatives and is substantially faster than existing methods that share information across time intervals. (SBIRR)

> In simulated and real data experiments, we find that our method consistently achieves nominal coverage, whereas all the alternatives dramatically fail to do so. (Smooth Sailing)

> In our experiments, we find that our approximation works well for a simple linear model. But we find that it can struggle in hierarchical models with more complex structure. (Sensitivity of MCMC)

> In the analyses below, we find that forecasts contain some useful information about air pollution, particularly in terms of mean excess exposure. For many smoke events, at least one forecast predicts elevated PM2.5 levels that correlate with the smoke event. However, no single forecast consistently predicts smoke events. Additionally, there are smoke events that none of the forecasts predict. (PM2.5)

Baselines: "We compare SnapMMD against two baselines: (1) Schrödinger bridge with iterative reference refinement (SBIRR) …" / "Ordinary Least Squares (OLS), Sandwich Estimator, Importance Weighted Least Squares (KDEIW), Generalized Least Squares (GLS), Gaussian Process Bayesian Credible Intervals (GP BCIs)" / "a persistence baseline" / "the method of Masoero et al. (2022)".

Per-experiment run-in: "**Vortex with zero divergence.** First, we consider a vortex with no divergence. The SE-Helmholtz GP is better at predicting the current and identifying the lack of divergence." Real-data opener: "We consider a part of the analysis of microcredit done in Meager [2022]. Originally, Meager studied …". Synthetic-as-check: "We first confirm that our method is able to predict well on data simulated from our own model …, as a check on our methodology before moving to real data."

---

## 9. Figure references and captions

References in text:
> In Fig. 1, we see that our method's forecast (red dots) is closer to ground truth than the baselines are. MMD (LV, Forecast in Table 1) and EMD (LV, Forecast in Table 2) agree that our method performs best.
> The second column in Figure 1 shows predictions from the SE-Helmholtz GP (upper) and SE-velocity GP (lower) at the test points.

Captions:
> Lotka-Volterra results (Section 4.1). We show 200 samples at each of 10 training times and 1 forecast time (red).
> First column: ground truth predictions (upper) and divergence (lower). Second column: current predictions. Third column: divergence estimates. Fourth column: posterior divergence z-values.
> Coverages (left) and confidence interval widths (right) for our method as well as 5 other methods. Only our method and GP BCIs consistently achieve the nominal coverage rate (95%) in this example; the GP BCIs line (dashed blue) overlaps with ours, in solid black, for most of the shifts. Of the two methods with correct coverage, our method yields much narrower intervals.
> Air quality measurements obtained from AirNow (orange), a persistence baseline (black, dashed line) and 6 forecasts (gray, varying markers) in Philadelphia during June 4–9, 2023.
> (Linear model) Histogram of treatment effect MCMC draws. The blue line indicates the sample mean. The dashed red line is the zero threshold. The dotted blue lines indicate estimates of approximate credible interval's endpoints.
> Left: Linear fit of mortality on admission heart rate for each group when holding all other covariates fixed at their group means. Right: Percentage of parameter space leading to sign flips.

---

## 10. Discussion and limitations

**SnapMMD** (restatement → **Limitations.** → **Social impact.**):

> In this work, we introduced a new method for learning SDEs from population-level snapshot data. Our approach is based on matching state—time distributions using a least squares scheme in a distributional space. Our proposed method handles real-life challenges such as unknown and state-dependent volatility, missing dimensions, and diagnostics for performance. Overall, our experiments indicate that our proposed framework outperforms existing methods in a wide range of applications.
>
> **Limitations.** While many scientific applications (including those above) feature useful models, it can nonetheless be a limitation that our method requires specification of an appropriate model family. A poorly chosen model family can lead to bad performance. An overly wide family can face identifiability challenges. In fact, even the complete time series of marginal distributions need not always uniquely determine the drift and volatility functions of the SDE; see Appendix E for further discussion. Finally, our method is not simulation-free, so it requires compute that scales with sample size and state dimension; development of a simulation-free method is an interesting future direction.
>
> **Social impact.** Our method might prove useful in areas of positive social impact, e.g. predicting a cell's reaction to a drug treatment or predicting the trajectory of oil particles in the ocean.

**Helmholtz GP** (a modelling choice defended as standard before being flagged):

> We have demonstrated the conceptual and empirical advantages of our Helmholtz GP relative to the velocity GP. A number of challenges remain. While we have focused on the purely spatial case for modularity, in general we expect currents to change over time, and it remains to extend our method to the spatiotemporal case – which we believe should be straightforward. … While our independent Gaussian noise model is standard in spatiotemporal modeling – and shared by Gonçalves et al. (2019); Lodise et al. (2020) – our real-data experimentation suggests that a better noise model might account for short-term wind oscillations and other noise patterns distinctive to oceans.

**SBIRR** (enumerated future work, with the honest "it is unclear"):

> Several interesting directions remain for future work. First, determining the precise conditions under which a latent SDE is identifiable from marginal samples remains an open problem; see appendix G for further discussion. Second, while our current method treats the volatility γ as fixed and constant across each iteration, one could introduce an outer loop that iteratively re-estimates γ from the data. … However, it is unclear how this extra loop would affect overall convergence, since each update to γ changes the underlying SB optimization problem. This is an interesting direction for future research. Additionally, our iterative refinement procedure may converge to suboptimal local minima. In practice, incorporating multiple random initializations or performing sensitivity analyses could help assess the robustness of the solution, and we plan to explore these aspects in future work.

**Smooth Sailing** (short, ends on a concrete extension):

> We propose a new method for constructing confidence intervals for spatial linear models. We show via theory and experiments that our intervals accurately quantify uncertainty under misspecification and covariate shift. In experiments, our method is the only method that consistently achieves (or even comes close to) nominal coverage. We observe that, very commonly in spatial data analyses, covariates and responses may be observed at different locations in space. Since our method does not actually use the source covariate values in inference for $\theta^{\star}_{\text{OLS}}$, it can be applied in this common missing-data scenario. Though it requires additional work, we believe the ideas here will extend naturally to the widely used class of generalized linear models.

**PM2.5** (honest negative finding in the same register as positive ones):

> We find that, when used to plan when to go outside during a high-pollution day, many forecasts (especially GEOS-CF, CAMS, and NAQFC) can substantially cut excess exposure to PM2.5. However, for deciding whether to go outside today, our analysis suggests that using the PM2.5 value in the morning would guide an individual about as well as, or better than, any forecast we analyzed here. We believe there remains meaningful room for improvement, and we hope that the present paper can help establish a framework for future evaluations as forecasts continue to improve.

**Oaxaca–Blinder** ("We here demonstrate"; future work in one sentence):

> It is well known that the OBD depends on the choice of reference group. We here demonstrate that this asymmetry can result in substantively different conclusions between the two reference choices. We present examples from real datasets as existence proofs. … Future work can determine to what extent our theoretical findings hold for non-uniform parameter distributions, other measures of model complexity, and nonlinear models.

---

## 11. Practical questions with a named actor

> For instance, an individual deciding whether to go outside at all today would like to reliably know whether PM2.5 levels will be high today. An individual choosing the best time to go outside would like to reliably know when PM2.5 levels will be lowest. (PM2.5)

> We imagine someone who needs to spend an hour outside, is flexible about when to go outside, wants to minimize PM2.5 exposure, and will use the morning forecast to make their decision. (PM2.5)

> Next, if a policymaker wants to advocate against microcredit deployment outside of Mexico, they need to know if microcredit remains detrimental beyond the data gathered in Angelucci et al. [2015]. (Sensitivity of MCMC)

> A practitioner with a particular definition of 'small data' can set α to reflect their concern. … A user might check approximation quality by dropping a proposed subset and re-running MCMC. (Sensitivity of MCMC)

> That is: "Will the top rankings from LLM-evaluation platforms change upon dropping a very small fraction of the human (or AI) preference evaluations?" A positive answer would raise concerns about the stability and generalizability of rankings produced by such systems. (LLM rankings)
