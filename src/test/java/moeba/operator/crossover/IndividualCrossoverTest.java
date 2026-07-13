package moeba.operator.crossover;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import moeba.operator.crossover.individual.rowcolbinary.impl.RowColUniformCrossover;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.uma.jmetal.util.binarySet.BinarySet;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

class IndividualCrossoverTest {

    @Test
    void uniformCrossoverSwapsEachBitSelectedByTheMaskOnce() {
        BinarySet first = binarySet(1, 0, 1, 0);
        BinarySet second = binarySet(0, 1, 0, 1);
        Random random = Mockito.mock(Random.class);
        Mockito.when(random.nextBoolean()).thenReturn(true, false, true, false);

        new RowColUniformCrossover(random).execute(first, second);

        assertEquals(binarySet(0, 0, 0, 0), first);
        assertEquals(binarySet(1, 1, 1, 1), second);
    }

    @Test
    void uniformCrossoverRejectsDifferentLengthBinarySets() {
        RowColUniformCrossover crossover = new RowColUniformCrossover(Mockito.mock(Random.class));

        assertThrows(
            IllegalArgumentException.class,
            () -> crossover.execute(new BinarySet(2), new BinarySet(3))
        );
    }

    @Test
    void defaultUniformCrossoverUsesTheConfiguredJMetalSeed() {
        JMetalRandom random = JMetalRandom.getInstance();
        long previousSeed = random.getSeed();
        try {
            BinarySet firstRunFirst = binarySet(1, 0, 1, 0, 1, 0, 1, 0);
            BinarySet firstRunSecond = binarySet(0, 1, 0, 1, 0, 1, 0, 1);
            random.setSeed(91234L);
            new RowColUniformCrossover().execute(firstRunFirst, firstRunSecond);

            BinarySet secondRunFirst = binarySet(1, 0, 1, 0, 1, 0, 1, 0);
            BinarySet secondRunSecond = binarySet(0, 1, 0, 1, 0, 1, 0, 1);
            random.setSeed(91234L);
            new RowColUniformCrossover().execute(secondRunFirst, secondRunSecond);

            assertEquals(firstRunFirst, secondRunFirst);
            assertEquals(firstRunSecond, secondRunSecond);
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
