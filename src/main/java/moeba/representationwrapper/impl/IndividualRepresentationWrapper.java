package moeba.representationwrapper.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.operator.crossover.individual.IndividualCrossover;
import moeba.operator.crossover.individual.rowcolbinary.RowColBinaryCrossover;
import moeba.operator.crossover.individual.rowcolbinary.impl.RowColUniformCrossover;
import moeba.operator.mutation.individual.IndividualMutation;
import moeba.operator.mutation.individual.rowcolbinary.RowColBinaryMutation;
import moeba.operator.mutation.individual.rowcolbinary.impl.RowColUniformMutation;
import moeba.representationwrapper.RepresentationWrapper;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.util.binarySet.BinarySet;

public class IndividualRepresentationWrapper extends RepresentationWrapper {
    private static final String DEFAULT_CROSSOVER_OPERATOR = "RowColUniformCrossover";
    private static final String DEFAULT_MUTATION_OPERATOR = "RowColUniformMutation";
    private static final Set<Class<? extends FitnessFunction>> SUPPORTED_FITNESS_FUNCTION_TYPES =
        Set.<Class<? extends FitnessFunction>>of(IndividualBiclusterFitnessFunction.class);

    public IndividualRepresentationWrapper(int numRows, int numColumns) {
        super(numRows, numColumns);
    }

    public CompositeSolution buildComposition(IntegerSolution integerSolution, BinarySolution binarySolution) {
        return new CompositeSolution(Arrays.asList(integerSolution, binarySolution));
    }

    @Override
    public int getNumIntVariables() {
        return 0;
    }

    @Override
    public int getNumBinaryVariables() {
        return 1;
    }

    @Override
    public int getLowerIntegerBound() {
        return 0;
    }

    @Override
    public int getUpperIntegerBound() {
        return 0;
    }

    @Override
    public int getNumBitsPerVariable() {
        return super.numRows + super.numColumns;
    }

    @Override
    public String getSummariseMethod() {
        return null;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ArrayList<ArrayList<Integer>[]> getBiclustersFromRepresentation(CompositeSolution solution) {
        // Initialize the result list
        ArrayList<ArrayList<Integer>[]> res = new ArrayList<>();
        
        // Extract binary variable from the composite solution
        BinarySet bs = ((BinarySolution) solution.variables().get(1)).variables().get(0);
        
        // Initialize bicluster, rows and cols
        ArrayList<Integer>[] bicluster = new ArrayList[2];
        ArrayList<Integer> rows = new ArrayList<>();
        ArrayList<Integer> cols = new ArrayList<>();

        // Iterate over rows and add them to the bicluster if they are set to true
        for (int i = 0; i < super.numRows; i++) {
            if (bs.get(i)) {
                rows.add(i);
            }
        }
        
        // Iterate over columns and add them to the bicluster if they are set to true
        for (int i = 0; i < super.numColumns; i++) {
            if (bs.get(i + super.numRows)) {
                cols.add(i);
            }
        }

        // Add rows and cols to the bicluster and add it to the result list
        bicluster[0] = rows;
        bicluster[1] = cols;
        res.add(bicluster);
        
        return res;
    }

    @Override
    public String[] getVarLabels() {
        String[] varLabels = new String[super.numRows + super.numColumns];
        for (int i = 0; i < super.numRows; i++) {
            varLabels[i] = "R" + i;
        }
        for (int i = 0; i < super.numColumns; i++) {
            varLabels[i + super.numRows] = "C" + i;
        }
        return varLabels;
    }

    @Override
    public String getDefaultCrossoverOperator() {
        return DEFAULT_CROSSOVER_OPERATOR;
    }

    @Override
    public String getDefaultMutationOperator() {
        return DEFAULT_MUTATION_OPERATOR;
    }

    @Override
    public Set<Class<? extends FitnessFunction>> getSupportedFitnessFunctionTypes() {
        return SUPPORTED_FITNESS_FUNCTION_TYPES;
    }

    @Override
    public CrossoverOperator<CompositeSolution> getCrossoverFromString(String strCrossoverOperator, double crossoverProbability, int numApproxCrossovers) {
        CrossoverOperator<CompositeSolution> res;
        String[] listStrCrossovers = strCrossoverOperator.split(";");

        if (listStrCrossovers.length == 1) {
            RowColBinaryCrossover rowColBinaryCrossover = getRowColBinaryCrossoverFromString(listStrCrossovers[0]);
            res = new IndividualCrossover(crossoverProbability, rowColBinaryCrossover);
        } else {
            throw new RuntimeException("The number of crossover operators is not supported for INDIVIDUAL representation.");
        }
        
        return res;
    }

    @Override
    public MutationOperator<CompositeSolution> getMutationFromString(String strMutationOperator, String mutationProbability, int numApproxMutations) {
        MutationOperator<CompositeSolution> res;
        String[] listStrMutations = strMutationOperator.split(";");

        if (listStrMutations.length == 1) {
            RowColBinaryMutation rowColBinaryMutation = getRowColBinaryMutationFromString(listStrMutations[0]);
            res = new IndividualMutation(mutationProbability, numApproxMutations, rowColBinaryMutation);
        } else {
            throw new RuntimeException("The number of mutation operators is not supported for INDIVIDUAL representation.");
        }
        
        return res;
    }

    public RowColBinaryCrossover getRowColBinaryCrossoverFromString(String str) {
        RowColBinaryCrossover res;
        switch (str.toLowerCase()) {
            case "rowcoluniformcrossover":
                res = new RowColUniformCrossover();
                break;
            default:
                throw new RuntimeException(
                        "The row col binary crossover " + str + " is not implemented.");
        }
        return res;
    }

    public RowColBinaryMutation getRowColBinaryMutationFromString(String str) {
        RowColBinaryMutation res;
        switch (str.toLowerCase()) {
            case "rowcoluniformmutation":
                res = new RowColUniformMutation();
                break;
            default:
                throw new RuntimeException(
                        "The row col binary mutation " + str + " is not implemented.");
        }
        return res;
    }
    
}
