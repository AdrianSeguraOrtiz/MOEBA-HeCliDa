package moeba.constraint;

import java.util.ArrayList;

/**
 * Evaluates a constraint over the biclusters represented by an individual.
 * Non-negative values are feasible and negative values represent violations,
 * following jMetal's constraint convention.
 */
@FunctionalInterface
public interface ConstraintFunction {
    double run(ArrayList<ArrayList<Integer>[]> biclusters);
}
