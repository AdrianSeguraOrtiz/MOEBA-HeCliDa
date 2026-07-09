package moeba.representationwrapper.impl;

import java.util.ArrayList;
import java.util.Arrays;

import moeba.representationwrapper.RepresentationWrapper;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;

public class SpecificRepresentationWrapper extends RepresentationWrapper {
    private int numBiclusters;
    private String summariseIndividualObjectives;

    public SpecificRepresentationWrapper(int numRows, int numColumns, int numBiclusters, String summariseIndividualObjectives) {
        super(numRows, numColumns);
        if (numBiclusters < 2 || numBiclusters >= super.numRows) {
            throw new IllegalArgumentException("The number of biclusters must be between 2 and " + (super.numRows - 1) + ".");
        }
        this.numBiclusters = numBiclusters;
        this.summariseIndividualObjectives = summariseIndividualObjectives;
    }

    public CompositeSolution buildComposition(IntegerSolution integerSolution, BinarySolution binarySolution) {
        return new CompositeSolution(Arrays.asList(integerSolution, binarySolution));
    }

    @Override
    public int getNumIntVariables() {
        return super.numRows;
    }

    @Override
    public int getNumBinaryVariables() {
        return numBiclusters;
    }

    @Override
    public int getLowerIntegerBound() {
        return 0;
    }

    @Override
    public int getUpperIntegerBound() {
        return numBiclusters - 1;
    }

    @Override
    public int getNumBitsPerVariable() {
        return super.numColumns;
    }

    @Override
    public String getSummariseMethod() {
        return summariseIndividualObjectives;
    }

    @Override
    public ArrayList<ArrayList<Integer>[]> getBiclustersFromRepresentation(CompositeSolution solution) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getBiclustersFromRepresentation'");
    }

    @Override
    public String[] getVarLabels() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getVarLabels'");
    }

    @Override
    public String getDefaultCrossoverOperator() {
        throw new IllegalArgumentException("No default crossover operator is configured for SPECIFIC representation. Use --crossover-operator explicitly.");
    }

    @Override
    public String getDefaultMutationOperator() {
        throw new IllegalArgumentException("No default mutation operator is configured for SPECIFIC representation. Use --mutation-operator explicitly.");
    }

    @Override
    public CrossoverOperator<CompositeSolution> getCrossoverFromString(String strCrossoverOperator,
            double crossoverProbability, int numApproxCrossovers) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getCrossoverFromString'");
    }

    @Override
    public MutationOperator<CompositeSolution> getMutationFromString(String strMutationOperator,
            String mutationProbability, int numApproxMutations) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getMutationFromString'");
    }
}
