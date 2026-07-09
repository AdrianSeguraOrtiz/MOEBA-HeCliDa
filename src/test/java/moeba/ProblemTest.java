package moeba;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import moeba.representationwrapper.impl.GenericRepresentationWrapper;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import org.junit.jupiter.api.Test;

class ProblemTest {
    private final double[][] data = new double[][] {
        {0.5, 0.2, 0.7, 0.3},
        {0.1, 0.4, 0.6, 0.8},
        {0.7, 0.8, 0.5, 0.1},
        {0.2, 0.3, 0.8, 0.6}
    };
    private final Class<?>[] numericTypes = new Class<?>[] {Float.class, Float.class, Float.class, Float.class};

    @Test
    void individualRepresentationAcceptsIndividualBiclusterObjectives() {
        assertDoesNotThrow(() -> new Problem(
            data,
            numericTypes,
            new String[] {"BiclusterSizeNormComp(summariseIndividualObjectives=Mean,rowsWeight=0.5)"},
            null,
            null,
            new IndividualRepresentationWrapper(4, 4)
        ));
    }

    @Test
    void individualRepresentationRejectsGenericBiclusterObjectives() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new Problem(
                data,
                numericTypes,
                new String[] {"DistanceBetweenBiclustersNormComp"},
                null,
                null,
                new IndividualRepresentationWrapper(4, 4)
            )
        );

        assertTrue(exception.getMessage().contains("DistanceBetweenBiclustersNormComp"));
        assertTrue(exception.getMessage().contains("GenericBiclusterFitnessFunction"));
        assertTrue(exception.getMessage().contains("IndividualRepresentationWrapper"));
    }

    @Test
    void individualRepresentationRejectsGlobalObjectivesBeforeConstructingThem() {
        Class<?>[] nonNumericTypes = new Class<?>[] {String.class, String.class, String.class, String.class};

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new Problem(
                data,
                nonNumericTypes,
                new String[] {"RegulatoryCoherenceNormComp"},
                null,
                null,
                new IndividualRepresentationWrapper(4, 4)
            )
        );

        assertTrue(exception.getMessage().contains("RegulatoryCoherenceNormComp"));
        assertTrue(exception.getMessage().contains("GlobalFitnessFunction"));
        assertTrue(exception.getMessage().contains("IndividualRepresentationWrapper"));
    }

    @Test
    void multiBiclusterRepresentationAcceptsGenericBiclusterObjectives() {
        assertDoesNotThrow(() -> new Problem(
            data,
            numericTypes,
            new String[] {"DistanceBetweenBiclustersNormComp(summariseIndividualObjectives=Mean)"},
            null,
            null,
            new GenericRepresentationWrapper(4, 4, 0.05f, 0.25f, "Mean")
        ));
    }
}
