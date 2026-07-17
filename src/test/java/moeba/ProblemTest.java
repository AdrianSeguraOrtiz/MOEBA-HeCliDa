package moeba;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import moeba.constraint.ConstraintFunction;
import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.impl.BiclusterSizeNormComp;
import moeba.fitnessfunction.impl.DistanceBetweenBiclustersNormComp;
import moeba.representationwrapper.impl.GenericRepresentationWrapper;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import moeba.utils.observer.ProblemObserver;
import moeba.utils.storage.impl.LocalCache;
import org.junit.jupiter.api.Test;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;

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
    void problemAcceptsInstantiatedFitnessFunctions() {
        Problem problem = new Problem(
            new FitnessFunction[] {new BiclusterSizeNormComp(data, numericTypes, null, null, 0.5)},
            null,
            null,
            new IndividualRepresentationWrapper(4, 4)
        );

        assertEquals(1, problem.getNumberOfObjectives());
        assertTrue(problem.getFitnessFunctions()[0] instanceof BiclusterSizeNormComp);
    }

    @Test
    void problemEvaluatesAndStoresProgrammaticConstraints() {
        ConstraintFunction constraint = biclusters -> 0.25;
        Problem problem = constrainedProblem(constraint, null);
        CompositeSolution solution = problem.createSolution();

        problem.evaluate(solution);

        assertEquals(1, problem.getNumberOfConstraints());
        assertEquals(1, solution.constraints().length);
        assertEquals(0.25, solution.constraints()[0]);
        for (Solution<?> component : solution.variables()) {
            assertEquals(1, component.constraints().length);
        }
        ConstraintFunction[] firstCopy = problem.getConstraintFunctions();
        ConstraintFunction[] secondCopy = problem.getConstraintFunctions();
        assertNotSame(firstCopy, secondCopy);
        assertEquals(constraint, firstCopy[0]);
    }

    @Test
    void constraintsAreEvaluatedWhenObjectivesComeFromExternalCache() {
        AtomicInteger constraintEvaluations = new AtomicInteger();
        ConstraintFunction constraint = biclusters -> {
            constraintEvaluations.incrementAndGet();
            return 0.5;
        };
        LocalCache<String, Double[]> cache = new LocalCache<>();
        Problem problem = constrainedProblem(constraint, cache);
        CompositeSolution solution = problem.createSolution();
        selectAllBits(solution);

        problem.evaluate(solution);
        problem.evaluate(solution);

        assertEquals(2, constraintEvaluations.get());
        assertEquals(1, cache.getNumGetters());
    }

    @Test
    void cachedObjectiveVectorIsValidatedBeforeUse() {
        LocalCache<String, Double[]> cache = new LocalCache<>();
        Problem problem = constrainedProblem(biclusters -> 0.5, cache);
        CompositeSolution solution = problem.createSolution();
        selectAllBits(solution);
        String key = StaticUtils.biclustersToString(
            problem.getRepresentationWrapper().getBiclustersFromRepresentation(solution)
        );
        cache.put(key, new Double[0]);

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> problem.evaluate(solution)
        );

        assertTrue(exception.getMessage().contains("exactly 1 values"));
    }

    @Test
    void problemRejectsNullAndNonFiniteConstraints() {
        assertThrows(
            IllegalArgumentException.class,
            () -> constrainedProblem(null, null)
        );
        Problem problem = constrainedProblem(biclusters -> Double.NaN, null);

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> problem.evaluate(problem.createSolution())
        );

        assertTrue(exception.getMessage().contains("Constraint at index 0 must be finite"));
    }

    @Test
    void problemRejectsInstantiatedFitnessFunctionsIncompatibleWithRepresentation() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new Problem(
                new FitnessFunction[] {new DistanceBetweenBiclustersNormComp(data, numericTypes, null, null)},
                null,
                null,
                new IndividualRepresentationWrapper(4, 4)
            )
        );

        assertTrue(exception.getMessage().contains("DistanceBetweenBiclustersNormComp"));
        assertTrue(exception.getMessage().contains("IndividualBiclusterFitnessFunction"));
        assertTrue(exception.getMessage().contains("IndividualRepresentationWrapper"));
    }

    @Test
    void problemObserverAcceptsInstantiatedFitnessFunctions() {
        assertDoesNotThrow(() -> new ProblemObserver(
            new FitnessFunction[] {new BiclusterSizeNormComp(data, numericTypes, null, null, 0.5)},
            null,
            null,
            new IndividualRepresentationWrapper(4, 4),
            new ProblemObserver.ObserverInterface[0]
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

    private Problem constrainedProblem(
        ConstraintFunction constraint,
        LocalCache<String, Double[]> cache
    ) {
        return new Problem(
            new FitnessFunction[] {
                new BiclusterSizeNormComp(data, numericTypes, null, null, 0.5)
            },
            new ConstraintFunction[] {constraint},
            cache,
            null,
            new IndividualRepresentationWrapper(4, 4)
        );
    }

    private static void selectAllBits(CompositeSolution solution) {
        BinarySolution binary = (BinarySolution) solution.variables().get(1);
        binary.variables().get(0).set(0, binary.variables().get(0).getBinarySetLength());
    }
}
