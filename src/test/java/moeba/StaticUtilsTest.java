package moeba;

import org.testng.annotations.Test;
import java.io.BufferedWriter;
import java.io.FileWriter;
import static moeba.StaticUtils.csvToObjectMatrix;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.testng.Assert.assertEquals;

public class StaticUtilsTest {

    @Test
    public void testCsvToObjectMatrix() throws IOException {
        // Create a temporary CSV file with a header
        File inputDataset = File.createTempFile("testCsvToObjectMatrix", ".csv");
        inputDataset.deleteOnExit();
        BufferedWriter bw = new BufferedWriter(new FileWriter(inputDataset));
        bw.write("header1,header2,header3\n");
        bw.write("value11,value12,value13\n");
        bw.write("value21,value22,value23\n");
        bw.close();
        // Read the CSV file and check the resulting array
        Object[][] matrix = csvToObjectMatrix(inputDataset);
        assertEquals(2, matrix.length);
        assertEquals(3, matrix[0].length);
        assertEquals("value11", matrix[0][0]);
        assertEquals("value12", matrix[0][1]);
        assertEquals("value13", matrix[0][2]);
        assertEquals("value21", matrix[1][0]);
        assertEquals("value22", matrix[1][1]);
        assertEquals("value23", matrix[1][2]);
    }

    @Test
    public void testJsonToClassArray() throws IOException {
        // Create a temporary JSON file with column types
        File inputJsonFile = File.createTempFile("testJsonToClassArray", ".json");
        inputJsonFile.deleteOnExit();
        BufferedWriter bw = new BufferedWriter(new FileWriter(inputJsonFile));
        bw.write("{\"column1\":\"string\",\"column2\":\"int\"}");
        bw.close();

        // Define column names
        String[] columnNames = {"column1", "column2"};

        // Call the method under test
        Class<?>[] columnClasses = StaticUtils.jsonToClassArray(inputJsonFile, columnNames);

        // Verify the result
        assertEquals(columnClasses.length, 2);
        assertEquals(columnClasses[0], String.class);
        assertEquals(columnClasses[1], Integer.class);
    }

    @Test
    public void testJsonToClassArray_ColumnNotFound() throws IOException {
        // Create a temporary JSON file with column types
        File inputJsonFile = File.createTempFile("testJsonToClassArray_ColumnNotFound", ".json");
        inputJsonFile.deleteOnExit();
        BufferedWriter bw = new BufferedWriter(new FileWriter(inputJsonFile));
        bw.write("{\"column1\":\"string\",\"column2\":\"int\"}");
        bw.close();

        // Define column names with a non-existing column
        String[] columnNames = {"column1", "column2", "column3"};

        // Call the method under test and expect an IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> {
            StaticUtils.jsonToClassArray(inputJsonFile, columnNames);
        });
    }

    @Test
    public void testJsonToClassArray_UnsupportedType() throws IOException {
        // Create a temporary JSON file with column types
        File inputJsonFile = File.createTempFile("testJsonToClassArray_UnsupportedType", ".json");
        inputJsonFile.deleteOnExit();
        BufferedWriter bw = new BufferedWriter(new FileWriter(inputJsonFile));
        bw.write("{\"column1\":\"string\",\"column2\":\"unsupported_type\"}");
        bw.close();

        // Define column names
        String[] columnNames = {"column1", "column2"};

        // Call the method under test and expect an IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> {
            StaticUtils.jsonToClassArray(inputJsonFile, columnNames);
        });
    }

    @Test
    public void testGetBiclustersFromRepresentation() {
        Integer[] x = {5,0,3,1,2,7,6,4, 1,0,1,0,0,1,0,0, 1,0,0,1,1,1,0,0, 1,1,1,0,0,0,0,0, 0,1,1,0,0,0,1,1, 0,0,0,1,1,1,0,0, 0,0,0,1,1,1,0,0, 0,0,0,0,0,0,1,1, 1,0,0,0,0,0,1,1, 0,1,1,0,0,0,1,1};
        ArrayList<ArrayList<Integer>[]> res = StaticUtils.getBiclustersFromRepresentation(x, Representation.GENERIC, 8, 8);

        // Num of biclusters
        assertEquals(4, res.size());

        // Create array
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] expected = new ArrayList[2];

        // Bicluster 1
        expected[0] = new ArrayList<Integer>(Arrays.asList(5));
        expected[1] = new ArrayList<Integer>(Arrays.asList(0,1,6));
        assertEquals(expected[0], res.get(0)[0]);
        assertEquals(expected[1], res.get(0)[1]);

        // Bicluster 2
        expected[0] = new ArrayList<Integer>(Arrays.asList(0,3));
        expected[1] = new ArrayList<Integer>(Arrays.asList(1,2,7));
        assertEquals(expected[0], res.get(1)[0]);
        assertEquals(expected[1], res.get(1)[1]);

        // Bicluster 3
        expected[0] = new ArrayList<Integer>(Arrays.asList(1,2,7));
        expected[1] = new ArrayList<Integer>(Arrays.asList(0,3,4));
        assertEquals(expected[0], res.get(2)[0]);
        assertEquals(expected[1], res.get(2)[1]);

        // Bicluster 4
        expected[0] = new ArrayList<Integer>(Arrays.asList(6,4));
        expected[1] = new ArrayList<Integer>(Arrays.asList(2,5,6,7));
        assertEquals(expected[0], res.get(3)[0]);
        assertEquals(expected[1], res.get(3)[1]);
    }
}