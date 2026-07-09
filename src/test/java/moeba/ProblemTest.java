package moeba;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

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
    private final ColumnType[] numericTypes = new ColumnType[] {
        ColumnType.numeric(),
        ColumnType.numeric(),
        ColumnType.numeric(),
        ColumnType.numeric()
    };
    private final ColumnType[] mixedTypes = new ColumnType[] {
        ColumnType.numeric(),
        ColumnType.categoricalNominal(),
        ColumnType.bool(),
        ColumnType.categoricalOrdinal(Arrays.asList("low", "high"))
    };

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
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new Problem(
                data,
                mixedTypes,
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

    @Test
    void structuralObjectivesAcceptMixedColumnTypes() {
        assertDoesNotThrow(() -> new Problem(
            data,
            mixedTypes,
            new String[] {"BiclusterSizeNormComp(summariseIndividualObjectives=Mean,rowsWeight=0.5)"},
            null,
            null,
            new IndividualRepresentationWrapper(4, 4)
        ));
    }

    @Test
    void numericObjectivesRejectNonNumericColumnTypes() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new Problem(
                data,
                mixedTypes,
                new String[] {"BiclusterVarianceNorm"},
                null,
                null,
                new IndividualRepresentationWrapper(4, 4)
            )
        );

        assertTrue(exception.getMessage().contains("BiclusterVarianceNorm"));
        assertTrue(exception.getMessage().contains("CATEGORICAL_NOMINAL"));
        assertTrue(exception.getMessage().contains("NUMERIC"));
    }

    @Test
    void globalNumericObjectivesUseCentralizedColumnTypeValidation() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new Problem(
                data,
                mixedTypes,
                new String[] {"RegulatoryCoherenceNormComp"},
                null,
                null,
                new GenericRepresentationWrapper(4, 4, 0.05f, 0.25f, "Mean")
            )
        );

        assertTrue(exception.getMessage().contains("RegulatoryCoherenceNormComp"));
        assertTrue(exception.getMessage().contains("CATEGORICAL_NOMINAL"));
        assertTrue(exception.getMessage().contains("NUMERIC"));
    }
}
