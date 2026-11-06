# Manuscript organization review and revision plan

Reviewed 5 November 2026: the saved `London urban cooling-ms-0806.docx` in
the author's OneDrive submission folder. Three independent AI reviewers took
health-policy, scientific-coherence and senior-editor perspectives. These were
simulated scholarly reviews, not external peer review or a Nature Health decision.
The Word manuscript was read, not edited; its SHA-256 before and after review was
`90d6ee4f032817a584cb49a856e163fdb6e3467db42d80285aa65b48d27cbacd`.

## Recommendation

Retain the multi-outcome architecture and most existing prose. The organizing
chain is **land-use choices → cooling → mortality, energy and work-capacity
outcomes → distribution of health benefits**. Health-benefit equity is not an
analysis of energy or occupational equity. Use “modelled heavy-work capacity,”
not observed productivity or economic output.

The health-policy reviewer initially favored one combined Green–Target section,
but changed that recommendation after reading the full manuscript. All three
now support two short consecutive health-distribution sections, sharing the
combined Figure 6, with Figure 7 adding distinct neighborhood evidence.

| Perspective | Main contribution to the recommendation |
|---|---|
| Health-policy | Keep all three co-benefits; move original Target findings out of Discussion. |
| Scientific coherence | Preserve the common scenario comparison, but distinguish aggregate outcomes from health-specific equity. |
| Senior editor | Fix the Results–Discussion boundary and incompatible numbers before stylistic polishing. |

This is a targeted reorganization, not a full rewrite. Nature Health's
[scope](https://www.nature.com/naturehealth/aims) emphasizes originality,
timeliness and policy/practice relevance; it does not prescribe this sequence.
Submission format and display-item limits still require a final current check.

## Proposed Results structure

| Section | Minimal change to existing text | Figures |
|---|---|---|
| Heat exposure and baseline inequalities across London | Rename “Exposure to heat under climate change”; retain baseline SVI/canopy context and shorten the future-temperature catalogue. | 1–2 |
| Land-use choices modify cooling and its health and economic co-benefits | Retain the existing cooling → citywide outcomes → borough variation sequence. Lead the co-benefit paragraph with mortality, followed by energy and work capacity; no separate outcome sections needed. | 3–5 |
| General greening yields uneven neighborhood health benefits | Preserve socioeconomic then older-population paragraphs; limit conclusions to Green and briefly report intensity consistency. | Green curves in 6; S1 |
| Targeting redistributes benefits across vulnerability dimensions | Follow Green with Target GAM contrasts, then fixed-threshold neighbourhood shares and the older-population trade-off. | 7; S2; combined 6 if selected |

Figure numbering is provisional. Do not automatically remove Figure 5: retain
it if it adds meaningful cross-outcome spatial information; otherwise consider
Extended Data. Preserve energy and work capacity regardless of map placement.

## Exact Results–Discussion move

1. Under Discussion → “Spatial targeting shapes the equity outcomes of urban
   greening,” locate “These within-city patterns mirror global evidence…”.
2. Keep its literature comparisons and proposed explanations in Discussion.
3. Move/replace the original findings beginning “In contrast, when greening is
   explicitly targeted…” and “Under these targeted greening scenarios,
   adaptation benefits shift…” into the new fourth Results section, immediately
   after the existing uneven-health-benefits section and before Discussion.
4. Do not copy those claims unchanged: add the verified older-population
   trade-off and retain descriptive, neighborhood-level interpretation.
5. Use one paragraph for Figure 6's gradient contrasts and a second for Figure
   7's paired neighbourhood redistribution, now checked against saved outputs.
6. Move Figure 7's presentation (currently before “Uncertainty, limitations and
   future work” in Discussion) alongside those Results.
7. Keep the idea beginning “These findings suggest that the distribution of
   adaptation benefits is not an inherent property…” in Discussion, qualified
   by competing vulnerability priorities. Avoid repeating numeric results there.

Use the [Figure 6 review pack](FIGURE6_REVIEW_DRAFTS.md) for its legends and
Green Results, and the [Figure 7 review](FIGURE7_REVIEW.md) for its legend and
Target Results. These linked drafts retain separate Green and Target sections.
Do not duplicate those drafts here. The combined candidate and S1–S2 are local
outputs, not figures already inserted in the reviewed Word manuscript.

## Other surgical edits

- Introduction: in “Here we address…”, introduce the matched-added-canopy
  Green–Target design and distinguish multi-outcome assessment from mortality
  distribution. Keep the broader energy and work-capacity framing.
- Discussion: condense its repeated opening synthesis and introductory heat
  statistics under “Land-use remains a modifiable lever…”. Retain the co-benefits
  subsection and the targeting interpretation, not first presentation of results.
- Move the detailed stewardship/aftercare paragraph to policy implications and
  shorten the final recap.
- Methods/Table 1: describe Target alongside Green; clarify the mortality
  comparator, SVI construction, allocation, GAMs and spatial sensitivity.

## Reconciliation gates before submission

These are inconsistencies in the reviewed manuscript, not verified replacement
numbers. Trace each claim to its output, scenario coverage, units and time basis.

- Abstract £22.4 million and 347–640 deaths versus Results £0.53 million and
  59–377 deaths. Distinguish point estimates from uncertainty bounds.
- July versus annual energy reporting; do not annualize a warm month by ×12.
- Six-scenario table/legends versus the nine-scenario analysis including Target.
- Four GCMs in Results versus three in Methods/Figure 1.
- Per-100,000 health Methods versus per-million Figure 6.
- Ambiguous July 2021 comparator versus fixed-mid-century land-use comparisons.
- Legacy centered-effect Extended Data Figure 3 versus full-response Target
  predictions; avoid presenting both as interchangeable.

## Next steps and boundaries

1. Review and approve the completed [Figure 7 audit and v4 candidate](FIGURE7_REVIEW.md).
2. Review the main Figure 6 candidate and S1–S2 in the linked review pack.
3. Review the linked Green and Target Results together; drafts are complete,
   but author approval and manuscript insertion remain pending.
4. Reconcile manuscript numbers and definitions with outputs.
5. Apply the limited structural changes only after author approval.

Figure changes, Word edits, model reruns and GitHub pushes are not authorized
by this document. Subsequent authorized work produced revised figures and prose;
the Word manuscript remains unchanged. The [TODO](../../TODO.md) remains the task register; the
[older revision brief](MANUSCRIPT_REVISION_BRIEF.md) preserves historical context.
