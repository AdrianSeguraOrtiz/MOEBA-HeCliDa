package moeba;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import moeba.representationwrapper.impl.GenericRepresentationWrapper;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import moeba.representationwrapper.impl.SpecificRepresentationWrapper;
import org.junit.jupiter.api.Test;

class RunnerDefaultsTest {

    @Test
    void genericRepresentationDefaultsUseGenericOperators() {
        GenericRepresentationWrapper wrapper = new GenericRepresentationWrapper(10, 5, 0.05f, 0.25f, "Mean");

        assertEquals(
            "GroupedBasedCrossover;CellUniformCrossover",
            wrapper.getDefaultCrossoverOperator()
        );
        assertEquals(
            "SwapMutation;BicUniformMutation;CellUniformMutation",
            wrapper.getDefaultMutationOperator()
        );
    }

    @Test
    void individualRepresentationDefaultsUseIndividualOperators() {
        IndividualRepresentationWrapper wrapper = new IndividualRepresentationWrapper(10, 5);

        assertEquals(
            "RowColUniformCrossover",
            wrapper.getDefaultCrossoverOperator()
        );
        assertEquals(
            "RowColUniformMutation",
            wrapper.getDefaultMutationOperator()
        );
    }

    @Test
    void observerDefaultsDependOnRepresentationAndEnabledCaches() {
        assertEquals(
            "FitnessEvolutionMinObserver;BiclusterCountObserver",
            Runner.getDefaultObservers(Representation.GENERIC, false, false)
        );
        assertEquals(
            "FitnessEvolutionMinObserver",
            Runner.getDefaultObservers(Representation.INDIVIDUAL, false, false)
        );
        assertEquals(
            "FitnessEvolutionMinObserver;ExternalCacheObserver;InternalCacheObserver",
            Runner.getDefaultObservers(Representation.INDIVIDUAL, true, true)
        );
    }

    @Test
    void unsupportedRepresentationsMustSpecifyOperatorsExplicitly() {
        SpecificRepresentationWrapper wrapper = new SpecificRepresentationWrapper(10, 5, 3, "Mean");

        assertThrows(
            IllegalArgumentException.class,
            wrapper::getDefaultCrossoverOperator
        );
        assertThrows(
            IllegalArgumentException.class,
            wrapper::getDefaultMutationOperator
        );
    }
}
