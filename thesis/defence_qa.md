# Questions and Answers

**Q1. PRIORITY. You assigned asset bands with a heuristic, why trust any coherence number?**

Ref: Section 3.1, Section 3.6, slide 9.

It is the headline caveat. The rule is hierarchical and conservative (subcategory metadata first, volatility quartiles only for untagged stocks, Balanced as default), flagged as a limitation, with external validation as future work.

**Q2. Why a one-band tolerance?**

Ref: Section 3.2.1, slide 9.

It mirrors the MiFID II adjacency allowance for borderline suitability.

**Q3. Why symmetric absolute distance, not asymmetric?**

Ref: Section 3.2.1.

The scale is ordinal, so absolute gap is natural, and I kept it symmetric for a clean first metric. The U-shaped per-band result already surfaces the asymmetry without baking direction in.

**Q4. Why a harmonic mean and a logistic ROI transform with tau equal to 0.01?**

Ref: Section 5.2, slides 20 and 24.

The harmonic mean collapses if any one axis is sacrificed, which punishes the usual trade-offs. ROI can be negative, so the logistic maps it to zero-to-one with flat return at 0.5 and tau setting one percentage point per month as the scale. Per-axis numbers are reported separately, so nothing hides inside the Balance.

**Q5. Could the 18.6% discordance be transaction-level noise?**

Ref: Section 3.4, slide 10.

No. The per-customer share is sharply bimodal (64.4% fully coherent, 17.2% fully discordant) and Hartigan's dip test rejects unimodality (D equal to 0.086, p less than 10^-15), so it is a persistent customer-level trait.

**Q6. PRIORITY. Adjusted R-squared is about 0.05, does that invalidate the +2.94 pp premium?**

Ref: Section 3.5, slides 12 and 32.

No. R-squared measures how much return variance the model explains, not whether the premium is real, and low R-squared is expected when idiosyncratic price moves dominate. The +2.94 pp premium is an average difference, not a prediction, and it is estimated precisely (p equal to 2.3 x 10^-13) with customer-clustered standard errors. Variance explained and effect significance are different questions.

**Q7. Is the premium causal?**

Ref: Section 3.5, Section 3.6, slide 12.

It is associational (identifies predictive patterns in observational data, how they correlate). It survives controls (raw gap +3.31 pp, conditioned +2.94 pp, so volatility, segment and year absorb about 11%). A difference-in-differences is the path to causation.

**Q8. The 2018 spike, you hypothesise but never test it.**

Ref: Section 3.4, Section 3.6, slide 11.

Correct, it is untested. MiFID II took force on 2018-01-03, the exact start of FAR-Trans, so a settling-in effect is plausible, but testing needs a pre-MiFID II counterfactual data which I do not have.

**Q9. Why LightGCN, not Random Forest or a sequential model?**

Ref: Section 4.2, slide 14.

LightGCN is the strongest ranking model and is differentiable with per-customer embeddings, so I can condition on the band and attach the margin term to BPR. Random Forest has no customer-level signal to route on. Other backbones are named future work.

**Q10. How are unbanded customers handled, why Balanced?**

Ref: Section 3.2.2, Section 4.2.

Only 320 of 29,090 (1.1%) lack a band. They are dropped from coherence (no band, no score) and fall back to the Balanced sub-model at inference as the neutral midpoint.

**Q11. Why lower median with conservative tie-breaks for regrouping?**

Ref: Section 4.3.2, slide 18.

The median resists outlier purchases, and resolving ties to the more conservative band is the prudent default for suitability. You would rather understate than overstate tolerance.

**Q12. The margin loss does not help, why include it?**

Ref: Section 4.3.1, Section 6.2, Section 6.4.

It is the control that sharpens the claim. Showing the model lever costs accuracy proves the data lever gain is real and not generic to any suitability pressure.

**Q13. Are the segmented models tuned fairly?**

Ref: Section 5.3, Section 5.4, Section 7.

They inherit the baseline best hyperparameters and are not re-tuned, by design, so the comparison isolates the lever not a tuning edge. This makes the suitability gains a lower bound and the accuracy costs an upper bound.

**Q14. 69 biweekly splits, is it leakage-safe on the temporal axis?**

Ref: Section 5.1, slide 21.

The cadence follows the original benchmark. Each split trains up to its cutoff and tests forward, deduplicated and filtered to the training-asset universe, so the temporal split is clean.

**Q15. Why an expanding window, not a rolling one, shouldn't the training start advance too?**

Ref: Section 5.1, slide 21.

No, the start is anchored and only the cutoff advances. Training is a customer-asset interaction graph, not a regime-sensitive price model, so early affinities stay valid and more history gives a denser graph. Dropping old data would also shrink the test universe, which is filtered to training assets, and worsen cold-start. A fixed-width rolling window is a clean drift-robustness ablation, named as future work.

**Q16. Your best model still has negative ROI (minus 0.006 per month), why acceptable?**

Ref: Section 6.1, Section 6.2, Section 5.2.

ROI@10 is roughly flat, not a meaningful loss, and the realistic baseline is already slightly negative, so no cost to return is the honest framing. The contribution is adding suitability and accuracy without paying in return.

**Q17. How do you know nDCG 0.330 to 0.352 is not averaging noise?**

Ref: Section 6.4, slide 25.

Per-split, regrouping wins nDCG on 100% of splits, PC on 100%, Balance on 71%. The margin loss wins Balance on 26% and nDCG on 7%, so effects are uniform shifts not lucky splits.

**Q18. Both baselines skew to higher-risk assets, why a defect not a preference?**

Ref: Section 6.1, slide 26.

Per-band lift rises monotonically and Conservative sits below random (LightGCN 0.75, Random Forest 0.52), so they serve cautious customers worse than chance. With no band-aware objective this is an uncontrolled byproduct of training on discordant purchases.
