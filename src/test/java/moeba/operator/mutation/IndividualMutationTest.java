package moeba.operator.mutation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import moeba.operator.mutation.individual.rowcolbinary.impl.RowColUniformMutation;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.uma.jmetal.util.binarySet.BinarySet;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

class IndividualMutationTest {

    @Test
    void uniformMutationWithZeroProbabilityDoesNotMutate() {
        BinarySet solution = binarySet(1, 0, 1, 0);
        Random random = Mockito.mock(Random.class);
        Mockito.when(random.nextDouble()).thenReturn(0.0);

        new RowColUniformMutation(random).execute(solution, 0.0);

        assertEquals(binarySet(1, 0, 1, 0), solution);
    }

    @Test
    void uniformMutationWithOneProbabilityMutatesEveryBit() {
        BinarySet solution = binarySet(1, 0, 1, 0);
        Random random = Mockito.mock(Random.class);
        Mockito.when(random.nextDouble()).thenReturn(0.99);

        new RowColUniformMutation(random).execute(solution, 1.0);

        assertEquals(binarySet(0, 1, 0, 1), solution);
    }

    @Test
    void uniformMutationMutatesBitsByIndependentProbability() {
        BinarySet solution = binarySet(1, 0, 1, 0);
        Random random = Mockito.mock(Random.class);
        Mockito.when(random.nextDouble()).thenReturn(0.4, 0.6, 0.2, 0.8);

        new RowColUniformMutation(random).execute(solution, 0.5);

        assertEquals(binarySet(0, 0, 0, 0), solution);
    }

    @Test
    void uniformMutationRejectsInvalidProbability() {
        RowColUniformMutation mutation = new RowColUniformMutation(Mockito.mock(Random.class));
        BinarySet solution = binarySet(1, 0, 1, 0);

        assertThrows(IllegalArgumentException.class, () -> mutation.execute(solution, -0.1));
        assertThrows(IllegalArgumentException.class, () -> mutation.execute(solution, 1.1));
        assertThrows(IllegalArgumentException.class, () -> mutation.execute(solution, Double.NaN));
    }

    @Test
    void defaultUniformMutationUsesTheConfiguredJMetalSeed() {
        JMetalRandom random = JMetalRandom.getInstance();
        long previousSeed = random.getSeed();
        try {
            BinarySet firstRun = binarySet(1, 0, 1, 0, 1, 0, 1, 0);
            random.setSeed(56789L);
            new RowColUniformMutation().execute(firstRun, 0.5);

            BinarySet secondRun = binarySet(1, 0, 1, 0, 1, 0, 1, 0);
            random.setSeed(56789L);
            new RowColUniformMutation().execute(secondRun, 0.5);

            assertEquals(firstRun, secondRun);
        } finally {
            random.setSeed(previousSeed);
        }
    }

    private static BinarySet binarySet(int... bits) {
        BinarySet binarySet = new BinarySet(bits.length);
        for (int i = 0; i < bits.length; i++) {
            binarySet.set(i, bits[i] == 1);
        }
        return binarySet;
    }
}
