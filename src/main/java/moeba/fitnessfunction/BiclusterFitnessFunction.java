package moeba.fitnessfunction;

import java.util.ArrayList;
import java.util.function.BiFunction;
import moeba.ColumnType;
import moeba.StaticUtils;
import moeba.utils.storage.CacheStorage;

public abstract class BiclusterFitnessFunction extends FitnessFunction {

    protected CacheStorage<String, Double> internalCache;

    public BiclusterFitnessFunction(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache, String summariseIndividualObjectives) {
        super(data, types);
        this.internalCache = internalCache;
        this.func = selectRunnableFunc(summariseIndividualObjectives);
    }

    private RunnableFunc selectRunnableFunc(String summariseMethod) {
        if (summariseMethod == null) summariseMethod = "mean";

        BiFunction<ArrayList<ArrayList<Integer>[]>, BiclusterScoreFunction, Double> summariser;
        switch (summariseMethod.toLowerCase()) {
            case "mean":
                summariser = this::calculateMean;
                break;
            case "harmonicmean":
                summariser = this::calculateHarmonicMean;
                break;
            case "geometricmean":
                summariser = this::calculateGeometricMean;
                break;
            default:
                throw new IllegalArgumentException("Summarise method not supported: " + summariseMethod);
        }
        return internalCache == null ? biclusters -> summariser.apply(biclusters, this::getBiclusterScore)
                                     : biclusters -> summariser.apply(biclusters, this::getCachedBiclusterScore);
    }

    @Override
    public double run(ArrayList<ArrayList<Integer>[]> biclusters) {
        return 1 - super.func.run(deleteSmallOrEmptyBiclusters(biclusters));
    }

    private double calculateMean(ArrayList<ArrayList<Integer>[]> biclusters, BiclusterScoreFunction scoreFunc) {
        return biclusters.stream()
                         .mapToDouble(bicluster -> scoreFunc.apply(biclusters, biclusters.indexOf(bicluster)))
                         .average()
                         .orElse(0);
    }

    private double calculateHarmonicMean(ArrayList<ArrayList<Integer>[]> biclusters, BiclusterScoreFunction scoreFunc) {
        return biclusters.size() / biclusters.stream()
                                             .mapToDouble(bicluster -> 1.0 / scoreFunc.apply(biclusters, biclusters.indexOf(bicluster)))
                                             .sum();
    }

    private double calculateGeometricMean(ArrayList<ArrayList<Integer>[]> biclusters, BiclusterScoreFunction scoreFunc) {
        return Math.pow(biclusters.stream()
                                  .mapToDouble(bicluster -> scoreFunc.apply(biclusters, biclusters.indexOf(bicluster)))
                                  .reduce(1, (a, b) -> a * b), 
                        1.0 / biclusters.size());
    }

    @FunctionalInterface
    private interface BiclusterScoreFunction {
        double apply(ArrayList<ArrayList<Integer>[]> biclusters, int i);
    }

    private double getCachedBiclusterScore(ArrayList<ArrayList<Integer>[]> biclusters, int i) {
        String key = StaticUtils.biclusterToString(biclusters.get(i));
        return internalCache.computeIfAbsent(key, k -> getBiclusterScore(biclusters, i));
    }

    protected abstract double getBiclusterScore(ArrayList<ArrayList<Integer>[]> biclusters, int i);
}
