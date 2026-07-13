package moeba.operator.crossover.individual.rowcolbinary.impl;

import java.util.Random;

import moeba.operator.crossover.individual.rowcolbinary.RowColBinaryCrossover;
import org.uma.jmetal.util.binarySet.BinarySet;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

public class RowColUniformCrossover implements RowColBinaryCrossover {
    private final Random random;

    public RowColUniformCrossover() {
        this(new Random(JMetalRandom.getInstance().nextInt(0, Integer.MAX_VALUE - 1)));
    }

    public RowColUniformCrossover(Random random) {
        this.random = random;
    }

    @Override
    public void execute(BinarySet s1, BinarySet s2) {
        int numBits = s1.getBinarySetLength();
        if (numBits != s2.getBinarySetLength()) {
            throw new IllegalArgumentException("Both binary sets must have the same length.");
        }

        boolean aux;
        for (int i = 0; i < numBits; i++) {
            if (random.nextBoolean()) {
                aux = s1.get(i);
                s1.set(i, s2.get(i));
                s2.set(i, aux);
            }
        }
    }
    
}
