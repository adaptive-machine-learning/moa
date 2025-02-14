package moa.classifiers.trees.sgtutil.neural;

import moa.classifiers.trees.sgtutil.GradHess;

public interface Layer {
    public GradHess[] update(double[] features, GradHess[] gradHess);

    public double[] predict(double[] features);
}