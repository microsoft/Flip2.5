---
name: paper-writing-renato
description: Draft, revise, and critique research papers (abstract, introduction, setup, method, theory, experiments, discussion, captions) in the style of Renato Berlinghieri, as learned from his PhD advisor Tamara Broderick at MIT (statistics / probabilistic ML). Also follows Jesse Shapiro's "four steps" process (aspirational intro → research → robot body → contractual intro). Use this whenever the user is working on a paper, manuscript, arXiv preprint, conference or journal submission, thesis chapter, abstract, contributions paragraph, theorem write-up, figure caption, limitations paragraph, or asks for feedback on academic prose — even if they only paste a LaTeX paragraph and say "does this read ok?" or "tighten this". Also use it when starting a new project ("help me think through the intro for a paper on X").
---

# Paper writing (Berlinghieri / Broderick style)

This skill encodes three things: (1) Shapiro's four-step process for getting from an idea to a first draft, (2) the concrete style visible in the group's recent papers (SnapMMD, Helmholtz GP, SBIRR, Smooth Sailing, PM2.5 forecasts, AMIP, Oaxaca–Blinder, LLM-ranking sensitivity, Double trouble), and (3) how to behave as a reviewer of the user's drafts. Read `references/exemplars.md` when you need to *see* the style (verbatim abstracts, intro paragraphs, theorem framing, captions) rather than be told about it; read `references/review-checklist.md` when critiquing a draft.

## 0. Figure out which mode you are in

- **Start mode** — no draft yet ("help me think about a paper on X"). Run Step 1 below: produce an *aspirational* abstract + contributions paragraph with made-up-but-plausible results, clearly marked, and ask whether the user would be happy with that paper.
- **Draft mode** — a section is requested. First collect the "contract" (Section 1), then write in the Berlinghieri/Broderick style (Sections 2–4). Never invent numbers, theorems, or baselines; leave `[TODO: …]` placeholders with a note of what the user must supply.
- **Review mode** — the user pastes text and wants feedback. Follow Section 5 and the checklist. Default to targeted edits that preserve the user's voice, not wholesale rewrites; the user has said explicitly that he wants minimal diffs and calm academic prose, and he pushes back on added complexity.
- **Polish mode** — the text is nearly done ("tighten", "cut 10%", "make this an 8-sentence abstract"). Apply Section 4 (sentence-level rules) and the abstract/intro skeletons; report what you cut and why in one or two lines.

Whatever the mode, keep the user's LaTeX macros, citation keys, and notation as they are.

## 1. Get the contract before you write

Every paper in this style can be summarised by four answers. Extract them from what the user gave you; if any is missing, ask for it (one question at a time) before drafting anything longer than a paragraph:

1. **Who wants what.** A scientist / practitioner / analyst and a concrete goal, in plain English. ("Oceanographers want to reconstruct currents away from buoys and find divergences.")
2. **What is natural, and why it falls short.** The obvious existing approach, described fairly and *named*, and the reasons it fails — ideally two, enumerated. ("An independent GP on each velocity component is natural … but (1) … (2) …")
3. **What we do, in one sentence of mechanism.** ("We put GP priors on the Helmholtz potentials instead of on the velocity components.")
4. **What the evidence is.** Theory (which guarantee), synthetic checks, real data, and the one headline number or blunt verdict — including negative findings.

These four answers *are* the abstract; the introduction expands each into a paragraph. Shapiro's test for whether the paper is worth writing: the question has an answer we do not already know, the answer is not obvious from logic or existing evidence, and knowing the answer would change what some practitioner does.

## 2. The process (Shapiro's four steps, adapted)

**Step 1 — Aspirational introduction.** Write the abstract and the contributions paragraph for the paper you *hope* to write, inventing results within reason and marking them `[aspirational]`. Ask the user: "If the paper delivers exactly this, are you happy?" If the imaginary results are not exciting, the real ones will not be either; stay in Step 1 and change the question, not the wording.

**Step 2 — Research.** The introduction is the compass. When the user asks what to do next, point them at the parts of the contract that are least certain (the guarantee that may not hold, the baseline that may win, the real dataset that may be messy), not the parts that are easy.

**Step 3 — Robot.** Write the body for a literal robot reader: *linear* (nothing is used before it is defined — notation, acronyms, assumptions, baselines), *clear* (every sentence has a referent the robot can resolve), *plain* (no fancy talk; the robot is not impressed), *formal* (mathematics is fine as long as it is correct). No persuasion in the body; persuasion lives in the introduction. Gaps found while writing send the user back to Step 2 — say so explicitly rather than papering over them.

**Step 4 — Contractual introduction.** Rewrite the abstract and introduction as a contract: the reader agrees to read carefully provided the paper delivers what the introduction promises. Build a promise ledger: for every claim in the abstract/intro, name the section, theorem, table, or figure that delivers it. A claim with no deliverer is either cut or sends the user back to Steps 2–3. When the paper delivers a contract the user is happy with, that is the first draft.

## 3. Berlinghieri/Broderick house structure, section by section

### 3.1 Abstract (≈ 8 sentences, ≈ 150–200 words; no citations, no equations, at most one acronym)

The group's abstracts fill the same slots in the same order:

1. *Who wants what*, plain English, no method named: "Scientists often want to …", "Given sparse observations of buoy velocities, oceanographers are interested in …", "Practitioners often aim to …".
2. *One concrete instance*: "For example, in time-course single-cell mRNA profiling, …" / "E.g., given single-cell sequencing data, …".
3. *The natural approach and why it is natural*: "Recent Schrödinger-bridge (SB) methods are natural for interpolating between snapshots." / "Gaussian processes (GPs) offer an attractive model."
4. *The turn*, opened by a bare **But** or **However**, with the failure enumerated: "But past SB papers have not addressed forecasting — likely since existing methods either (1) … or (2) …". A gap may be stated as an absence: "To the best of our knowledge, there has been no systematic investigation of …".
5. *"In the present work, we propose/show …"* (this exact phrase is the group's signature), or "So we propose a new method that (1) … and (2) …".
6. *Mechanism in one sentence*: what the method actually does.
7. *Contrast*: "Unlike past work, …" / "In contrast to classical methods that rely on an i.i.d. assumption, …".
8. *Evidence*: "We demonstrate … via both theory and experiments on synthetic and real data", plus the headline number or the honest verdict ("dropping just 0.003% of preferences can change the top-ranked model"; "the performance of our method is mixed"). Optional: extra deliverables (a diagnostic statistic, a new metric, the code URL).

One intensifier is permitted in the whole abstract ("surprisingly", "remarkably"). Negative results go in the abstract, stated flatly.

### 3.2 Introduction (≈ 5 paragraphs; no roadmap paragraph; contributions in prose)

| Shapiro's slots | How the group fills them |
|---|---|
| Motivation (why it matters) | ¶1: the practitioner and the goal, with "For example / For instance" within three sentences, often a named actor or a real study ("suppose a data scientist at Hospital H notices …"; "Consider this motivating example. Angelucci et al. [2015] …"). Key terms are defined inline in prose, not in math. |
| Challenges (why unanswered) | ¶2: the natural existing approach, cited generously, described fairly, and **given a name** ("We call such a model the *velocity GP*."). ¶3: the gap, opened by "However"/"But", "for two reasons: (1) … (2) …". The naive fix is raised and dismissed ("a brute-force search over all small subsets is computationally infeasible"). Priority claims are always qualified: "To our knowledge", "We are not aware of". |
| What we do, assumptions, approach | ¶4: "In the present work / To address these issues, we propose …" — mechanism in 3–5 sentences, then benefits enumerated *inside* a sentence: "Our formulation yields three key benefits: (1) …, (2) …, and (3) …". Each claim may carry its section inline: "(Section 3)". |
| Findings + contributions | ¶5: preview of the evidence in words ("Our primary focus is on forecasting, where … consistently outperforms …; on interpolation our method matches or exceeds … in almost all cases."), then "Code to reproduce the experiments is available at <URL>." |

Never a bulleted contributions list. The default is no roadmap paragraph — section numbers ride inline in the contributions paragraph ("(Section 3)"); when a venue expects one (the ICLR paper), it is two sentences at the very end of the intro, not "The rest of the paper is organized as follows". Related work is woven into ¶2–3 with packed citation lists, each prior work followed by one sentence of *precisely* what it does not do ("But these methods assume the field is known on a grid, whereas our goal is to reconstruct it from sparse observations."); the full discussion goes to a run-in "**Related work.**" paragraph after the intro or to an appendix ("Additional/Extended Related Work"), closed by "In sum, then, it is not clear from existing work that …". For a journal/applied paper a content-titled section ("Prior PM2.5 Forecast Evaluations") is used instead.

### 3.3 Setup / background

Open with the data and its indexing: "We consider a dataset $\mathcal{D}$ of $M$ observations $\{(\mathbf{x}_m, \mathbf{y}_m)\}_{m=1}^M$. Here $\mathbf{x}_m$ represents …". Every symbol gets a plain-language gloss in the same or the next sentence. Organise with bold run-in headings that are short noun phrases ending in a period: **Data.** **Goal.** **Model.** **Our estimand.** **Metrics of success.** Announce a deliberate terminology choice ("Though our work applies beyond scRNA-seq, we describe our data and goals using scRNA-seq terminology to concretize our notation."). "Recall that …" restates notation right before it is needed.

### 3.4 Method

Start by contrasting with the named baseline: "Instead of putting separate GP priors on the two velocity components as in the velocity GP, we propose to …". Method subsections are gerund/noun run-in headings: **Handling incomplete state measurements.** **Choosing the kernel.** Introduce the mechanism after a goal sentence with "To that end," / "To do so,"; introduce the precise list after a loose statement with "Namely," or "Precisely,". In running text the method is "our method" far more often than its proper name. Name new objects with "so we call it the *Most Influential Set*" (italic at first use; capitalised long name + acronym for the headline object).

### 3.5 Theory

The five-beat pattern around every formal statement:

1. One sentence, plain English, saying what will be shown and why it matters: "In fact, for any isotropic choice of kernels, we show that the velocity GP must assume the same prior variance on divergence and vorticity."
2. The statement, in "Let … / Take assumptions 2.1, 3.1 and 3.2 / Suppose … / Then …" syntax; a "That is, …" gloss inside the statement is fine. Label **Proposition** for structural facts, **Theorem** for the headline guarantee, and name it parenthetically by what it characterises: "Proposition 4.4 (OBD unexplained sign flips)."
3. Proof pointer as the *first* sentence after: "See Appendix D.1 for the proof." / "This result (proof in Appendix B.2) shows that …".
4. One sentence of interpretation, often with a numerical sanity check: "…so the error of the linear approximation shrinks faster than the function itself as α → 0 (e.g., when α = 0.01, √α = 0.1 ≫ 0.01)."
5. The contrast with the baseline: "The equivariance in Proposition 4.2 need not hold for velocity GP priors."

Assumptions are numbered and stated in their own environments *before* the theorem, each preceded by a one-line motivation ("Assumption 2 introduces the key simplification."; "it will be convenient to assume invertibility, as for OLS.") and given semi-formally with an "i.e." gloss. Theory sections may be titled as questions ("What determines AMIP robustness?") and open with a one-sentence roadmap ("First, in Section 4.1, we analyse …"). Counterexamples are flagged: "For intuition in the other direction, consider the following counterexample." In short-format venues keep definitions and an informal statement in the main text ("In Theorem B.1, we show that …") and push the formal result to the appendix. Limitations of the theory are stated where it lives: "One limitation of this theory is that it assumes …, when in practice we may expect …".

### 3.6 Experiments

Open with the verdict, before any setup — the group's formula is literally "In simulated and real experiments, we find that our method …", followed immediately by the honest caveat if there is one ("But we find that it can struggle in hierarchical models."). Then **Metrics of success.**, **Baselines.** (full descriptive name followed by acronym: "Schrödinger bridge with iterative reference refinement (SBIRR)"; always include a trivial baseline and name it — "a persistence baseline"; name datasets and prior methods by their source paper), then one run-in heading per experiment naming the scenario (**Vortex with zero divergence.**). Synthetic experiments come first and are described as a check ("as a check on our methodology before moving to real data"); real-data experiments replicate a published analysis and open with one background sentence about it.

Each experiment paragraph is *setup sentence → result sentence → exhibit pointer*, with the exhibit cited as evidence for a conclusion stated in words and the visual cue decoded: "In Fig. 1, we see that our method's forecast (red dots) is closer to ground truth than the baselines are." Explain one exhibit fully before introducing the next. Headline numbers are given as count and percentage together ("dropping just two (0.003% of) evaluations"). A "practical question" is a decision by a named person in a concrete situation ("We imagine someone who needs to spend an hour outside, is flexible about when, and will use the morning forecast").

Captions are self-contained and structural: name the experiment and section, say what is plotted with counts, decode the legend, optionally end with the one-line takeaway. "Lotka–Volterra results (Section 4.1). We show 200 samples at each of 10 training times and 1 forecast time (red)." / "Coverages (left) and interval widths (right) for our method and 5 others. … Of the two methods with correct coverage, ours yields much narrower intervals."

### 3.7 Discussion

One-sentence restatement ("In this work, we introduced …"), then a bold run-in **Limitations.** paragraph (or "Several directions remain for future work. First, … Second, …"). The limitation template is *concession → limitation → consequence → pointer or future direction*: "While many scientific applications feature useful model families, it can nonetheless be a limitation that our method requires specifying one. A poorly chosen family can lead to bad performance … see Appendix E. Finally, our method is not simulation-free, so its compute scales with sample size; a simulation-free variant is an interesting future direction." A modelling choice is defended as "standard … and shared by [prior work]" before being flagged. Optional **Social impact.** paragraph, one or two sentences, concrete.

## 4. Sentence-level rules

- **Voice.** Active first-person plural for what we do ("We show", "We find", "we expect", "we believe"); the practitioner is the grammatical subject of motivation sentences ("Scientists want …", "An analyst might conclude …", "One might think …"). Passive only for well-known facts. Sentences may start with **But**, **And**, **So**, **Yet**, **Then,** — this is a deliberate marker of the style, not an error to fix.
- **Rhythm.** Medium sentences (15–25 words) punctuated by very short ones ("The rough intuition is as follows." "How could we check this?"). Long sentences are long only because of an inline enumeration or a citation list, never because of nested clauses. Rhetorical question → immediate naive answer → why it fails.
- **Enumerate inline**, "(1) … (2) …" or "(i) … (ii) …" inside a sentence, or "First, … Second, … Finally, …" across sentences. No bullet lists in the paper body.
- **Parentheticals, not footnotes**, for glosses and hedges: "(chosen before seeing data)", "(and often better than)", "(by far)". Footnotes only for a simplifying convention, the code URL, a citation-backed justification of a convention, or an implementation aside — two to five per paper at most.
- **Hedges are quantified**, never vague: "in almost all experiments", "at least as good as (and often better than)", "about as well as, or better than", "likely since", "we expect". Miscalibration runs both ways: if the theorem and the tables support a stronger sentence, write the stronger sentence.
- **Preferred connectives**: "In particular," (narrow to the instance), "In practice," (what an analyst actually does), "That is," (restate plainly), "Note that" (a technical caveat, sparingly), "We emphasize that" (once per paper, for a choice a reader might mistake for a necessity), "To that end,", "Namely,", "Precisely,", "By contrast,", "In sum, then,", "Crucially," (once).
- **Naming.** Define acronyms expansion-first at first use ("maximum mean discrepancy (MMD)"); cross-reference as "Section 3", "Appendix E.2", "Theorem B.1", "Fig. 1"; use ":=" for definitions; write dropped fractions as "100α%" with a concrete instance "(e.g., α = 0.001)". Bold lowercase vectors, bold capitals for stacked matrices, calligraphic sets, stars for target/test quantities, hats for empirical quantities, subscript 0 for the true data-generating object.
- **Words to avoid**: "novel", "leverage", "utilize", "significantly" in the non-statistical sense, "state-of-the-art" for our own method, "it is worth noting", "delve", stacked intensifiers. The non-native construction "discuss about" → "discuss". Avoid enthusiasm markers and exclamation points; the register is calm.
- **Vague referents.** "the remaining X", "this", "these results" without an explicit noun; pronouns that only make sense to someone who already knows the paper; passives that hide the actor ("it was observed that") — name who or what.

## 5. Review mode: how to critique a draft

Read `references/review-checklist.md` for the full list. The behaviours that matter most:

- Run the **contract audit** first: list every promise in the abstract/intro and the section that delivers it; flag promises with no deliverer and deliverables that are never promised.
- Run the **robot pass** on the body: anything used before it is defined, any sentence whose referent the robot cannot resolve, any jargon doing no work, any mathematics that is not correct as written.
- Compress multi-sentence wind-ups into the shortest direct statement of what the paper does. Flag claims that are not self-evidently justified and *ask why* rather than polishing around them. Ask for the topic sentence when a paragraph lacks one; ask why content that belongs right after an earlier point is not there.
- Distinguish what was done from how it is described; precision about method matters even under a word limit.
- Anticipate the specific objection a reader will have later and suggest addressing or reordering it earlier.
- When several exhibits are introduced together, ask for one-at-a-time explanation. When a reader is told about an exhibit without a way to locate it, ask where to look. Flag named results or terms used without saying how the passage relates back to them.
- When setting-specific detail crowds out the methodological story, propose a concrete trim ("what do you lose if you keep only the first and last sentence?") or a move to a footnote/appendix.
- Name the specific worry ("a reader will lose track of which of the four estimators this refers to") rather than "this is confusing". When the same critique applies elsewhere, say so once. Mark a genuinely good equation, sentence, or exhibit with a short concrete affirmation at that spot.
- Watch both directions of miscalibration: push back on overclaiming *and* on unwarranted hedging, and ask the user to justify the extra caution.
- Deliver feedback as targeted edits (quote → replacement, with a one-line reason) rather than a rewritten section, unless the user asks for a rewrite.

If the user is struggling with clear writing in general, point them to the readings Shapiro cites: Cochrane, "Writing Tips for PhD Students" (2005); McCloskey, *Economical Writing* (1985); Thomson, *A Guide for the Young Economist*, ch. 1 (2001).

## 6. Bundled references

- `references/exemplars.md` — verbatim abstracts, intro paragraphs, theorem framings, experiment openers, captions, and limitations paragraphs from the group's papers, organised by section. Open it when drafting a section, and paste the closest exemplar next to your draft for comparison.
- `references/review-checklist.md` — the full reviewer behaviour list plus the contract-audit and robot-pass procedures, formatted to work through top to bottom.
