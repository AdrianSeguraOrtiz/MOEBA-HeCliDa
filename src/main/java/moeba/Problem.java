package moeba;

import java.util.ArrayList;
import java.util.Arrays;

import moeba.fitnessfunction.FitnessFunction;
import moeba.problem.AbstractMixedIntegerBinaryProblem;
import moeba.representationwrapper.RepresentationWrapper;
import moeba.utils.storage.CacheStorage;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.binarysolution.impl.DefaultBinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.solution.integersolution.impl.DefaultIntegerSolution;

/**
 * Extends AbstractMixedIntegerBinaryProblem to define a custom problem with both integer and binary solution components.
 * It supports different types of representations, for addressing different problem structures.
 */
public class Problem extends AbstractMixedIntegerBinaryProblem {

    private final FitnessFunction[] fitnessFunctions;
    protected final CacheStorage<String, Double[]> externalCache;
    protected final CacheStorage<String, Double>[] internalCaches;
    protected final RepresentationWrapper representationWrapper;
    private final EvaluateFunction evaluateFunction;

    private static final class ValidatedFitnessFunctions {
        private final FitnessFunction[] values;

        private ValidatedFitnessFunctions(FitnessFunction[] values) {
            this.values = values;
        }
    }

    public interface EvaluateFunction {
        public CompositeSolution evaluate(CompositeSolution solution, ArrayList<ArrayList<Integer>[]> biclusters);
    }

    public Problem(
        double[][] data, 
        ColumnType[] types,
        String[] strFitnessFunctions, 
        CacheStorage<String, Double[]> externalCache, 
        CacheStorage<String, Double>[] internalCaches,
        RepresentationWrapper representationWrapper
    ) {
        this(
            createFitnessFunctions(data, types, strFitnessFunctions, internalCaches, representationWrapper),
            externalCache,
            internalCaches,
            representationWrapper
        );
    }

    public Problem(
        FitnessFunction[] fitnessFunctions,
        CacheStorage<String, Double[]> externalCache,
        CacheStorage<String, Double>[] internalCaches,
        RepresentationWrapper representationWrapper
    ) {
        this(
            validateFitnessFunctions(fitnessFunctions, internalCaches, representationWrapper),
            externalCache,
            internalCaches,
            representationWrapper
        );
    }

    private Problem(
        ValidatedFitnessFunctions fitnessFunctions,
        CacheStorage<String, Double[]> externalCache,
        CacheStorage<String, Double>[] internalCaches,
        RepresentationWrapper representationWrapper
    ) {
        super(
            representationWrapper.getNumIntVariables(), 
            representationWrapper.getNumBinaryVariables(), 
            representationWrapper.getLowerIntegerBound(), 
            representationWrapper.getUpperIntegerBound(), 
            representationWrapper.getNumBitsPerVariable()
        );
        this.externalCache = externalCache;
        this.internalCaches = internalCaches;
        this.representationWrapper = representationWrapper;
        this.evaluateFunction = externalCache == null ? this::evaluateWithoutCache : this::evaluateWithCache;
        this.fitnessFunctions = fitnessFunctions.values;

        // Configure the problem's parameters
        setNumberOfVariables(2);
        setNumberOfObjectives(this.fitnessFunctions.length);
        setName("Problem");
    }

    /**
     * Evaluates a solution, updating its objectives based on the defined fitness functions.
     * This involves converting the solution representation to biclusters and applying the fitness functions.
     *
     * @param solution The CompositeSolution instance to be evaluated.
     * @return CompositeSolution The evaluated solution with updated objective values.
     */
    @Override
    public CompositeSolution evaluate(CompositeSolution solution) {
        ArrayList<ArrayList<Integer>[]> biclusters = representationWrapper.getBiclustersFromRepresentation(solution);
        return evaluateFunction.evaluate(solution, biclusters);
    }

    /**
     * Evaluates the solution without using the cache, directly applying the fitness functions.
     *
     * @param solution The CompositeSolution instance to be evaluated.
     * @param biclusters The biclusters obtained from the solution representation.
     * @return CompositeSolution The evaluated solution with updated objective values.
     */
    public CompositeSolution evaluateWithoutCache(CompositeSolution solution, ArrayList<ArrayList<Integer>[]> biclusters){
        // Apply each fitness function to the biclusters and update the solution objectives
        for (int i = 0; i < fitnessFunctions.length; i++){
            solution.objectives()[i] = fitnessFunctions[i].run(biclusters);
        }
        return solution;
    }

    /**
     * Evaluates the solution using the external cache to avoid recalculating known results.
     * 
     * @param solution The solution to evaluate.
     * @param biclusters The biclusters derived from the solution.
     * @return The evaluated solution with updated objectives, potentially leveraging cached values.
     */
    public CompositeSolution evaluateWithCache(CompositeSolution solution, ArrayList<ArrayList<Integer>[]> biclusters){
        String key = StaticUtils.biclustersToString(biclusters);
        if (externalCache.containsKey(key)){
            for (int i = 0; i < fitnessFunctions.length; i++){
                solution.objectives()[i] = externalCache.get(key)[i];
            }
        } else {
            solution = evaluateWithoutCache(solution, biclusters);
            Double[] scores = new Double[fitnessFunctions.length];
            for (int i = 0; i < fitnessFunctions.length; i++){
                scores[i] = solution.objectives()[i];
            }
            externalCache.put(key, scores);
        }

        return solution;
    }

    /**
     * Creates a new solution with the appropriate number of objectives and constraints, as well as the
     * correct integer and binary representation.
     *
     * @return A new composite solution with the correct representation.
     */
    @Override
    public CompositeSolution createSolution() {
        IntegerSolution integerSolution = new DefaultIntegerSolution(getNumberOfObjectives(), getNumberOfConstraints(), super.integerBounds);
        BinarySolution binarySolution = new DefaultBinarySolution(super.numBitsPerVariable, getNumberOfObjectives());

        return representationWrapper.buildComposition(integerSolution, binarySolution);
    }

    public FitnessFunction[] getFitnessFunctions() {
        return Arrays.copyOf(fitnessFunctions, fitnessFunctions.length);
    }

    public RepresentationWrapper getRepresentationWrapper() {
        return representationWrapper;
    }

    private static ValidatedFitnessFunctions createFitnessFunctions(
        double[][] data,
        ColumnType[] types,
        String[] strFitnessFunctions,
        CacheStorage<String, Double>[] internalCaches,
        RepresentationWrapper representationWrapper
    ) {
        if (strFitnessFunctions == null || strFitnessFunctions.length == 0) {
            throw new IllegalArgumentException("Problem requires at least one fitness function.");
        }
        if (internalCaches != null && internalCaches.length != strFitnessFunctions.length) {
            throw new IllegalArgumentException("Problem requires one internal cache per fitness function.");
        }

        FitnessFunction[] fitnessFunctions = new FitnessFunction[strFitnessFunctions.length];
        for (int i = 0; i < strFitnessFunctions.length; i++) {
            StaticUtils.ObjectiveDefinition objectiveDefinition = StaticUtils.getObjectiveDefinitionFromString(strFitnessFunctions[i]);
            representationWrapper.validateFitnessFunctionType(
                strFitnessFunctions[i],
                objectiveDefinition.getFitnessFunctionType()
            );
            fitnessFunctions[i] = objectiveDefinition.create(
                strFitnessFunctions[i],
                data,
                types,
                internalCaches == null ? null : internalCaches[i],
                representationWrapper.getSummariseMethod()
            );
        }
        return new ValidatedFitnessFunctions(fitnessFunctions);
    }

    private static ValidatedFitnessFunctions validateFitnessFunctions(
        FitnessFunction[] fitnessFunctions,
        CacheStorage<String, Double>[] internalCaches,
        RepresentationWrapper representationWrapper
    ) {
        if (fitnessFunctions == null || fitnessFunctions.length == 0) {
            throw new IllegalArgumentException("Problem requires at least one fitness function.");
        }
        if (internalCaches != null && internalCaches.length != fitnessFunctions.length) {
            throw new IllegalArgumentException("Problem requires one internal cache per fitness function.");
        }

        FitnessFunction[] copy = Arrays.copyOf(fitnessFunctions, fitnessFunctions.length);
        for (int i = 0; i < copy.length; i++) {
            if (copy[i] == null) {
                throw new IllegalArgumentException("Fitness function at index " + i + " is null.");
            }
            representationWrapper.validateFitnessFunctionType(
                copy[i].getClass().getSimpleName(),
                copy[i].getClass()
            );
        }
        return new ValidatedFitnessFunctions(copy);
    }
    
}
