package moeba.representationwrapper;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import moeba.representationwrapper.impl.GenericRepresentationWrapper;
import org.junit.jupiter.api.Test;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.binarysolution.impl.DefaultBinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.solution.integersolution.impl.DefaultIntegerSolution;
import org.uma.jmetal.util.binarySet.BinarySet;
import org.uma.jmetal.util.bounds.Bounds;

public class RepresentationWrapperTest {
    
    @Test
    public void testGetBiclustersFromGenericRepresentation() {
        int[] x = {5,0,3,1,2,7,6,4, 1,0,1,0,0,1,0,0, 0,1,1,0,0,1,0,1, 1,0,0,1,0,1,0,0, 1,0,0,1,1,0,1,0, 0,1,1,0,0,0,0,1, 0,1,1,0,0,0,0,1, 0,0,0,0,1,0,1,0, 0,0,0,0,1,1,1,0, 1,0,0,1,1,0,1,0};
        
        List<Bounds<Integer>> integerBounds = new ArrayList<>(8);
        for (int i = 0; i < 8; i++) {
            integerBounds.add(Bounds.create(0, 7));
        }
        IntegerSolution integerSolution = new DefaultIntegerSolution(1, 0, integerBounds);
        for (int i = 0; i < 8; i++) {
            integerSolution.variables().set(i, x[i]);
        }

        BinarySolution binarySolution = new DefaultBinarySolution(Arrays.asList(8,8,8,8,8,8,8,8,8), 1);
        for (int i = 0; i < 9; i++) {
            BinarySet bits = new BinarySet(8);
            for (int j = 0; j < 8; j++) {
                bits.set(j, x[8 + i*8 + j] == 1);
            }
            binarySolution.variables().set(i, bits);
        }
        
        GenericRepresentationWrapper wrapper = new GenericRepresentationWrapper(8,8,0.05f,0.2f,"Mean");
        ArrayList<ArrayList<Integer>[]> res = wrapper.getBiclustersFromRepresentation(new CompositeSolution(Arrays.asList(integerSolution, binarySolution)));

        // Num of biclusters
        assertEquals(4, res.size());

        // Create array
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] expected = new ArrayList[2];

        // Bicluster 1
        expected[0] = new ArrayList<Integer>(Arrays.asList(0,3));
        expected[1] = new ArrayList<Integer>(Arrays.asList(1,2,7));
        assertEquals(expected[0], res.get(0)[0]);
        assertEquals(expected[1], res.get(0)[1]);

        // Bicluster 2
        expected[0] = new ArrayList<Integer>(Arrays.asList(1,2,7));
        expected[1] = new ArrayList<Integer>(Arrays.asList(0,3,4));
        assertEquals(expected[0], res.get(1)[0]);
        assertEquals(expected[1], res.get(1)[1]);

        // Bicluster 3
        expected[0] = new ArrayList<Integer>(Arrays.asList(4,6));
        expected[1] = new ArrayList<Integer>(Arrays.asList(2,5,6,7));
        assertEquals(expected[0], res.get(2)[0]);
        assertEquals(expected[1], res.get(2)[1]);

        // Bicluster 4
        expected[0] = new ArrayList<Integer>(Arrays.asList(5));
        expected[1] = new ArrayList<Integer>(Arrays.asList(0,1,6));
        assertEquals(expected[0], res.get(3)[0]);
        assertEquals(expected[1], res.get(3)[1]);
    }

    /** 
    @Test
    public void testGetBiclustersFromGenericRepresentationWithMerge() {
        int[] x = {5,0,3,1,2,7,6,4, 1,0,1,0,0,1,0,0, 1,1,1,1,0,1,0,1, 1,0,0,1,0,1,0,0, 0,0,0,0,1,0,1,0, 0,1,1,0,0,0,0,1, 0,1,1,0,0,0,0,1, 0,0,0,0,1,0,1,0, 1,0,0,1,1,1,1,0, 0,0,0,0,1,0,1,0};
        
        List<Bounds<Integer>> integerBounds = new ArrayList<>(8);
        for (int i = 0; i < 8; i++) {
            integerBounds.add(Bounds.create(0, 7));
        }
        IntegerSolution integerSolution = new DefaultIntegerSolution(1, 0, integerBounds);
        for (int i = 0; i < 8; i++) {
            integerSolution.variables().set(i, x[i]);
        }

        BinarySolution binarySolution = new DefaultBinarySolution(Arrays.asList(8,8,8,8,8,8,8,8,8), 1);
        for (int i = 0; i < 9; i++) {
            BinarySet bits = new BinarySet(8);
            for (int j = 0; j < 8; j++) {
                bits.set(j, x[8 + i*8 + j] == 1);
            }
            binarySolution.variables().set(i, bits);
        }
        
        GenericRepresentationWrapper wrapper = new GenericRepresentationWrapper(8,8,0.05f,0.2f,"Mean");
        ArrayList<ArrayList<Integer>[]> res = wrapper.getBiclustersFromRepresentation(new CompositeSolution(Arrays.asList(integerSolution, binarySolution)));

        // Num of biclusters
        assertEquals(3, res.size());

        // Create array
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] expected = new ArrayList[2];

        // Bicluster 1
        expected[0] = new ArrayList<Integer>(Arrays.asList(0,3,5));
        expected[1] = new ArrayList<Integer>(Arrays.asList(0,1,6));
        assertEquals(expected[0], res.get(0)[0]);
        assertEquals(expected[1], res.get(0)[1]);

        // Bicluster 2
        expected[0] = new ArrayList<Integer>(Arrays.asList(1,2,7));
        expected[1] = new ArrayList<Integer>(Arrays.asList(0,3,4));
        assertEquals(expected[0], res.get(1)[0]);
        assertEquals(expected[1], res.get(1)[1]);

        // Bicluster 3
        expected[0] = new ArrayList<Integer>(Arrays.asList(4,6));
        expected[1] = new ArrayList<Integer>(Arrays.asList(2,5,6,7));
        assertEquals(expected[0], res.get(2)[0]);
        assertEquals(expected[1], res.get(2)[1]);
    }
    */
}
