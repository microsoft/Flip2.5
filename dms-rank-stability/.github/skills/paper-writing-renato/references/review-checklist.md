# Review checklist

Work through this top to bottom when the user asks for feedback on a draft. Output targeted edits (quote → replacement → one-line reason) grouped by section, followed by at most five "bigger questions" for the author. Do not rewrite whole sections unless asked.

## A. Contract audit (abstract + introduction vs. the body)

1. List every promise in the abstract and introduction: each "we show / we prove / we propose / we find / our method achieves …".
2. For each promise, name the deliverer: a section, theorem, proposition, table, or figure. A promise with no deliverer is flagged: either cut it or the author returns to research/writing.
3. List deliverables in the body that are never promised (a theorem nobody mentioned in the intro, a real-data experiment absent from the abstract). Suggest promoting them or explain why they stay unpromoted.
4. Check the four contract answers are all present and in the house order: who wants what → natural approach and its two failures → what we do (mechanism) → evidence and headline.
5. Check calibration in both directions. Overclaim: any "consistently", "always", "the first", "state-of-the-art" without a table or theorem behind it. Underclaim: a theorem or a clean sweep in the tables described with "may", "could potentially", "we hope". Ask the author to justify extra hedging exactly as you would an overclaim.
6. Priority claims ("the first", "no prior work") carry "To our knowledge".

## B. Robot pass (body only)

1. Linear: every symbol, acronym, assumption, baseline name, and dataset is defined before first use. Search for acronyms and check the expansion appears first.
2. Clear: every "this", "these", "it", "the remaining X", "the former/latter", "as above" has an unambiguous noun. Every named result or term used in a passage says how the passage relates back to it.
3. Plain: strike words that do no work ("novel", "leverage", "it is worth noting that", "in order to", "significantly" outside a test). Replace passives that hide the actor with the actor.
4. Formal: statements are stated correctly as written (quantifiers, conditions, what is random). A theorem that is correct only under an unstated assumption is a bug, not a style issue.
5. Any gap found here is reported as "this needs more work, not more prose", and the author is sent back to the relevant research step.

## C. Structure

1. Abstract: does it follow the 8-slot skeleton? No citations, no equations, at most one acronym, one intensifier. Ends with the headline or the honest verdict, not a generic claim of relevance.
2. Intro: opens on a practitioner and a goal with an example within three sentences; the existing approach is named; the gap is enumerated; contributions are prose with inline (1)(2)(3); no roadmap paragraph; ends with the code URL when there is code.
3. Each paragraph has a topic sentence. When it does not, ask directly what the topic sentence is.
4. Content that should logically follow an earlier point but appears later: ask why it is not placed there.
5. Anticipate the specific objection a reader will form at paragraph N and check whether it is addressed before or at N. Suggest reordering when it is only addressed later.
6. Structures that fan out into many sub-components (four estimators × three settings × two metrics): name the specific tracking burden and suggest a table, a naming convention, or a cut.
7. Setting-specific institutional detail that crowds out the methodological contribution: propose a concrete trim ("what do you lose if you keep only the first and last sentence here?") or a footnote/appendix move.
8. When the same critique applies to several sub-sections, say it once: "something similar applies to Sections 4.2 and 4.3."

## D. Theory

1. Plain-English sentence before each statement saying what it shows.
2. Assumptions numbered, each with a one-line motivation before it, and re-listed by number in the theorem ("Take assumptions 2.1, 3.1 and 3.2").
3. First sentence after the statement is the proof pointer; second is the interpretation in words; then the contrast with the baseline.
4. Proposition vs. Theorem labelling is consistent with importance; the headline guarantee is a Theorem.
5. Limitations of the theory are stated next to the theory ("One limitation of this theory is that it assumes …").

## E. Experiments and exhibits

1. The section opens with the verdict ("In simulated and real experiments, we find that …") and the honest caveat.
2. Baselines: full descriptive name then acronym; a trivial baseline is present and named; datasets and competitor methods are named by source paper.
3. Each experiment paragraph: setup sentence → result sentence → exhibit pointer with the visual cue decoded ("(red dots)").
4. Exhibits are introduced and explained one at a time. If several are introduced together, ask for one-at-a-time.
5. Every exhibit mention tells the reader where to look ("Fig. 2, upper row"; "Table 1, LV Forecast column").
6. Captions are self-contained: experiment + section, what is plotted with counts, legend decoded, optional takeaway line.
7. A result presented without an interpretive conclusion: ask the author to state the inference ("given X, the likely inference is Y").
8. Headline numbers appear as count and percentage together.

## F. Discussion

1. One-sentence restatement, then a **Limitations.** paragraph following concession → limitation → consequence → pointer/future direction.
2. Future-work items are enumerated ("First, … Second, …") and specific.
3. If length allows, the paper ends on a concrete takeaway or result, not a general claim of relevance.

## G. Sentence level

1. Compress multi-sentence wind-ups into "We show/study X …".
2. Hedges quantified ("in almost all experiments"), not vague ("often", "may").
3. Parentheticals for glosses; footnotes only for conventions, code URLs, citation-backed justifications, implementation asides.
4. Inline enumerations; no bullets in the body.
5. Sentence-initial But/And/So are fine; do not "correct" them.
6. Calm register: no exclamation points, no stacked intensifiers, one "surprisingly" per paper.
7. "discuss about" → "discuss"; "utilize" → "use"; "in order to" → "to".

## H. How to deliver the review

- Lead with the two or three things that matter most (usually contract-audit findings), then section-by-section edits.
- Quote the original, give the replacement, give the reason in one line. Preserve the author's voice, macros, and notation.
- Flag a claim you cannot justify with a question ("why is this true?") instead of polishing the prose around it.
- Mark genuinely good content — a key equation, a clean exhibit, a well-turned sentence — with a short concrete affirmation at that spot.
- If the author is struggling with clear writing generally, point to Cochrane (2005) "Writing Tips for PhD Students", McCloskey (1985) *Economical Writing*, and Thomson (2001) *A Guide for the Young Economist*, ch. 1 — the readings cited at the end of Shapiro's "Four Steps to an Applied Micro Paper".
