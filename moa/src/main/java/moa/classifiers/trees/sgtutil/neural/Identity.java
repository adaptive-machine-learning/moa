package moa.classifiers.trees.sgtutil.neural;

import moa.classifiers.trees.sgtutil.GradHess;

import java.io.Serializable;

public class Identity implements Serializable, Layer {
    private static final long serialVersionUID = 1583592862930132577L;

    public GradHess[] update(double[] features, GradHess[] gradHess) {
        return gradHess.clone();
    }

    public double[] predict(double[] features) {
        return features.clone();
    }
}