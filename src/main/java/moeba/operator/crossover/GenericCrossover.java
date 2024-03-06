package moeba.operator.crossover;

import java.util.List;
import java.util.ArrayList;

import moeba.operator.crossover.biclustersbinary.BiclusterBinaryCrossover;
import moeba.operator.crossover.cellbinary.CellBinaryCrossover;
import moeba.operator.crossover.rowpermutation.RowPermutationCrossover;

import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.errorchecking.Check;

public class GenericCrossover implements CrossoverOperator<CompositeSolution> {
    private double crossoverProbability;
    private int numRows;
    private int numCols;
    private RowPermutationCrossover rowPermutationCrossover;
    private BiclusterBinaryCrossover biclusterBinaryCrossover;
    private CellBinaryCrossover cellBinaryCrossover;
    private JMetalRandom random;  

    public GenericCrossover(double crossoverProbability, int numRows, int numCols, RowPermutationCrossover rowPermutationCrossover, BiclusterBinaryCrossover biclusterBinaryCrossover, CellBinaryCrossover cellBinaryCrossover) {
        this.crossoverProbability = crossoverProbability;
        this.numRows = numRows;
        this.numCols = numCols;
        this.rowPermutationCrossover = rowPermutationCrossover;
        this.biclusterBinaryCrossover = biclusterBinaryCrossover;
        this.cellBinaryCrossover = cellBinaryCrossover;
        this.random = JMetalRandom.getInstance();
    }

    @Override
    public List<CompositeSolution> execute(List<CompositeSolution> source) {
        Check.notNull(source);
        Check.that(source.size() == 2, "There must be two parents instead of " + source.size());

        List<CompositeSolution> offspring = new ArrayList<>();
        CompositeSolution offSpring1 = (CompositeSolution) source.get(0).copy();
        IntegerSolution offSpring1IntSol = (IntegerSolution) offSpring1.variables().get(0);
        BinarySolution offSpring1BinSol = (BinarySolution) offSpring1.variables().get(1);
        CompositeSolution offSpring2 = (CompositeSolution) source.get(1).copy();
        IntegerSolution offSpring2IntSol = (IntegerSolution) offSpring2.variables().get(0);
        BinarySolution offSpring2BinSol = (BinarySolution) offSpring2.variables().get(1);
        if (random.nextDouble(0, 1) <= this.crossoverProbability) {
            
            // Rows permutation crossover
            int[] parent1RowPerm = new int[numRows];
            int[] parent2RowPerm = new int[numRows];
            for (int i = 0; i < numRows; i++) {
                parent1RowPerm[i] = offSpring1IntSol.variables().get(i);
                parent2RowPerm[i] = offSpring2IntSol.variables().get(i);
            }
            int[][] offSpringRowPerm = rowPermutationCrossover.execute(parent1RowPerm, parent2RowPerm);
            for (int i = 0; i < numRows; i++) {
                offSpring1IntSol.variables().set(i, offSpringRowPerm[0][i]);
                offSpring2IntSol.variables().set(i, offSpringRowPerm[1][i]);
            }

            // Biclusters binary crossover
            biclusterBinaryCrossover.execute(offSpring1BinSol.variables().get(0), offSpring2BinSol.variables().get(0));


            // Cells binary crossover
            for (int i = 1; i < numCols + 1; i++) {
                cellBinaryCrossover.execute(offSpring1BinSol.variables().get(i), offSpring2BinSol.variables().get(i));
            }
        } 

        offspring.add(offSpring1);
        offspring.add(offSpring2);
        return offspring;
    }

    @Override
    public double getCrossoverProbability() {
        return crossoverProbability;
    }

    @Override
    public int getNumberOfRequiredParents() {
        return 2;
    }

    @Override
    public int getNumberOfGeneratedChildren() {
        return 2;
    }
    
}
