package moeba.operator.mutation.individual.rowcolbinary.impl;

import java.util.Random;

import moeba.operator.mutation.individual.rowcolbinary.RowColBinaryMutation;
import org.uma.jmetal.util.binarySet.BinarySet;

public class RowColUniformMutation implements RowColBinaryMutation {
    
    private Random random;

    public RowColUniformMutation() {
        this.random = new Random();
    }

    public RowColUniformMutation(Random random) {
        this.random = random;
    }

    @Override
    public void execute(BinarySet bs, double mutationProbability) {
        if (Double.isNaN(mutationProbability) || mutationProbability < 0.0 || mutationProbability > 1.0) {
            throw new IllegalArgumentException("Mutation probability must be between 0 and 1.");
        }

        int numBits = bs.getBinarySetLength();
        for (int i = 0; i < numBits; i++) {
            if (random.nextDouble() < mutationProbability) {
                bs.set(i, !bs.get(i));
            }
        }
    }
}
