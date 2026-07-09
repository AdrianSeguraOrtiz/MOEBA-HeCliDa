package moeba;

import java.util.ArrayList;

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

    protected double[][] data;
    private ColumnType[] types;
    private FitnessFunction[] fitnessFunctions;
    protected CacheStorage<String, Double[]> externalCache;
    protected CacheStorage<String, Double>[] internalCaches;
    protected RepresentationWrapper representationWrapper;
    private EvaluateFunction evaluateFunction;

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
        super(
            representationWrapper.getNumIntVariables(), 
            representationWrapper.getNumBinaryVariables(), 
            representationWrapper.getLowerIntegerBound(), 
            representationWrapper.getUpperIntegerBound(), 
            representationWrapper.getNumBitsPerVariable()
        );
        this.data = data;
        this.types = types;
        this.externalCache = externalCache;
        this.internalCaches = internalCaches;
        this.representationWrapper = representationWrapper;
        this.evaluateFunction = externalCache == null ? this::evaluateWithoutCache : this::evaluateWithCache;
        
        // Initialize fitness functions based on provided string identifiers
        this.fitnessFunctions = new FitnessFunction[strFitnessFunctions.length];
        for (int i = 0; i < strFitnessFunctions.length; i++) {
            StaticUtils.ObjectiveDefinition objectiveDefinition = StaticUtils.getObjectiveDefinitionFromString(strFitnessFunctions[i]);
            representationWrapper.validateFitnessFunctionType(
                strFitnessFunctions[i],
                objectiveDefinition.getFitnessFunctionType()
            );
            this.fitnessFunctions[i] = objectiveDefinition.create(
                strFitnessFunctions[i],
                this.data,
                this.types,
                internalCaches == null ? null : internalCaches[i],
                representationWrapper.getSummariseMethod()
            );
        }

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
    
}
