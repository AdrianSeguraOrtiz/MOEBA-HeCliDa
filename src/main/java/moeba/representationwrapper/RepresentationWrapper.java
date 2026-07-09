package moeba.representationwrapper;

import java.util.ArrayList;
import java.util.Set;

import moeba.fitnessfunction.FitnessFunction;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;

public abstract class RepresentationWrapper {

    protected int numRows;
    protected int numColumns;

    public RepresentationWrapper(int numRows, int numColumns) {
        this.numRows = numRows;
        this.numColumns = numColumns;
    }

    public abstract int getNumIntVariables();
    public abstract int getNumBinaryVariables();
    public abstract int getLowerIntegerBound();
    public abstract int getUpperIntegerBound();
    public abstract int getNumBitsPerVariable();

    public abstract CompositeSolution buildComposition(IntegerSolution integerSolution, BinarySolution binarySolution);

    public abstract ArrayList<ArrayList<Integer>[]> getBiclustersFromRepresentation(CompositeSolution solution);
    public abstract String getDefaultCrossoverOperator();
    public abstract String getDefaultMutationOperator();
    public abstract Set<Class<? extends FitnessFunction>> getSupportedFitnessFunctionTypes();
    public abstract CrossoverOperator<CompositeSolution> getCrossoverFromString(String strCrossoverOperator, double crossoverProbability, int numApproxCrossovers);
    public abstract MutationOperator<CompositeSolution> getMutationFromString(String strMutationOperator, String mutationProbability, int numApproxMutations);

    public abstract String[] getVarLabels();
    public abstract String getSummariseMethod();

    public boolean supportsFitnessFunctionType(Class<? extends FitnessFunction> fitnessFunctionType) {
        for (Class<? extends FitnessFunction> supportedType : getSupportedFitnessFunctionTypes()) {
            if (supportedType.isAssignableFrom(fitnessFunctionType)) {
                return true;
            }
        }
        return false;
    }

    public void validateFitnessFunctionType(String fitnessFunction, Class<? extends FitnessFunction> fitnessFunctionType) {
        if (!supportsFitnessFunctionType(fitnessFunctionType)) {
            throw new IllegalArgumentException(
                "Objective " + fitnessFunction + " (" + fitnessFunctionType.getSimpleName() + ") is not compatible with " +
                getClass().getSimpleName() + ". Supported objective types: " + getSupportedFitnessFunctionTypeNames() + "."
            );
        }
    }

    private String getSupportedFitnessFunctionTypeNames() {
        ArrayList<String> names = new ArrayList<>();
        for (Class<? extends FitnessFunction> supportedType : getSupportedFitnessFunctionTypes()) {
            names.add(supportedType.getSimpleName());
        }
        return String.join(", ", names);
    }
}
