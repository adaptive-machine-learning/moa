package moa.classifiers.trees;

import java.util.ArrayList;
import java.util.List;

import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Utils;

/**
 * Strict Very Fast Decision Tree (SVFDT)
 * Minimal modification over HoeffdingTree (VFDT)
 *
 * <p>This is a minimal modification of MOA's HoeffdingTree/VFDT that applies
 * extra statistical constraints before accepting a split.</p>
 */
public class SVFDT extends HoeffdingTree {

    private static final long serialVersionUID = 1L;

    /* =========================
     * SVFDT-specific Incremental statistics
     * ========================= */

    protected long entropyCount = 0;
    protected double meanH = 0.0;
    protected double m2H = 0.0;


    protected long igCount = 0;
    protected double meanIG = 0.0;
    protected double m2IG = 0.0;


    protected long nCount = 0;
    protected double meanN = 0.0;
    protected double m2N = 0.0;

    /* =========================
     * Override split decision
     * ========================= */

    @Override
    protected void attemptToSplit(ActiveLearningNode node,
                                  SplitNode parent,
                                  int parentIndex) {

        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplits =
                    node.getBestSplitSuggestions(splitCriterion, this);

            if (bestSplits.length < 2) {
                return;
            }

            AttributeSplitSuggestion best = bestSplits[bestSplits.length - 1];
            AttributeSplitSuggestion secondBest = bestSplits[bestSplits.length - 2];

            double hb = computeHoeffdingBound(
                    splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                    splitConfidenceOption.getValue(),
                    node.getWeightSeen()
            );

            double meritDiff = best.merit - secondBest.merit;

            boolean vfdtDecision =
                    (meritDiff > hb) || (hb < tieThresholdOption.getValue());

            if (vfdtDecision) {

                // SVFDT additional constraints
                if (svfdtCanSplit(node, best.merit)) {

                    // Original VFDT split happens here
                    SplitNode newSplit = new SplitNode(best.splitTest,
                            node.getObservedClassDistribution());

                    for (int i = 0; i < best.numSplits(); i++) {
                        newSplit.setChild(i, newLearningNode());
                    }

                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }

                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;

                } else {
                    // SVFDT blocks unnecessary split
                    // do nothing (node remains leaf)
                }
            }
        }
    }

    /* =========================
     * SVFDT constraint logic
     * ========================= */

    protected boolean svfdtCanSplit(ActiveLearningNode node, double bestIG) {

        double entropy = computeEntropy(node.getObservedClassDistribution());
        double n = node.getWeightSeen();

        // update entropy statistics
        double[] h = updateStatistics(entropyCount, meanH, m2H, entropy);

        entropyCount = (long) h[0];
        meanH = h[1];
        m2H = h[2];

        // update IG statistics
        double[] ig = updateStatistics(igCount, meanIG, m2IG, bestIG);

        igCount = (long) ig[0];
        meanIG = ig[1];
        m2IG = ig[2];

        // update N statistics
        double[] nn = updateStatistics(nCount, meanN, m2N, n);

        nCount = (long) nn[0];
        meanN = nn[1];
        m2N = nn[2];

        // warm-up
        if (entropyCount < 5) {
            return true;
        }

        double stdH = computeStd(entropyCount, m2H);
        double stdIG = computeStd(igCount, m2IG);
        double stdN = computeStd(nCount, m2N);

        boolean c1 = entropy >= (meanH - stdH);
        boolean c2 = bestIG >= (meanIG - stdIG);
        boolean c3 = n >= (meanN - stdN);

        return c1 && c2 && c3;
    }

    /* =========================
     * Utility methods
     * ========================= */

    protected double computeEntropy(double[] dist) {
        double sum = Utils.sum(dist);
        if (sum == 0) {
            return 0;
        }

        double entropy = 0.0;
        for (double d : dist) {
            if (d > 0) {
                double p = d / sum;
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }
        return entropy;
    }

    protected double[] updateStatistics(long count, double mean, double m2, double value) {
        count++;

        double delta = value - mean;
        mean += delta / count;

        double delta2 = value - mean;
        m2 += delta * delta2;

        return new double[] {count, mean, m2};
    }

    protected double computeStd(long count, double m2) {
        if (count < 2) {
            return 0.0;
        }
        return Math.sqrt(m2 / count);
    }
}
