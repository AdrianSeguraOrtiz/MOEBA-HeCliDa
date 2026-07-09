package moeba;

import org.junit.jupiter.api.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import static moeba.StaticUtils.csvToStringMatrix;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class StaticUtilsTest {

    @Test
    public void testcsvToStringMatrix() throws IOException {
        // Create a temporary CSV file with a header
        File inputDataset = File.createTempFile("testcsvToStringMatrix", ".csv");
        inputDataset.deleteOnExit();
        BufferedWriter bw = new BufferedWriter(new FileWriter(inputDataset));
        bw.write("header1,header2,header3\n");
        bw.write("value11,value12,value13\n");
        bw.write("value21,value22,value23\n");
        bw.close();
        // Read the CSV file and check the resulting array
        String[][] matrix = csvToStringMatrix(inputDataset);
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
        assertEquals(2, columnClasses.length);
        assertEquals(String.class, columnClasses[0]);
        assertEquals(Integer.class, columnClasses[1]);
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
    public void testDataToNumericMatrixConvertsAllSupportedTypes() {
        String[][] data = {
            {"1", "2.5", "3.5", "Yes", "red"},
            {"2", "4.5", "5.5", "No", "blue"},
            {"3", "6.5", "7.5", "yes", "red"}
        };
        Class<?>[] types = {Integer.class, Double.class, Float.class, Boolean.class, String.class};

        double[][] numericData = StaticUtils.dataToNumericMatrix(data, types, 3);

        assertArrayEquals(new double[] {1.0, 2.5, 3.5, 1.0, 0.0}, numericData[0]);
        assertArrayEquals(new double[] {2.0, 4.5, 5.5, 0.0, 1.0}, numericData[1]);
        assertArrayEquals(new double[] {3.0, 6.5, 7.5, 1.0, 0.0}, numericData[2]);
    }

    @Test
    public void testDataToNumericMatrixRejectsInvalidThreadCount() {
        String[][] data = {{"1"}};
        Class<?>[] types = {Integer.class};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 0));
    }

    @Test
    public void testDataToNumericMatrixRejectsMismatchedTypes() {
        String[][] data = {{"1", "2"}};
        Class<?>[] types = {Integer.class};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 1));
    }

    @Test
    public void testDataToNumericMatrixRejectsRowsWithDifferentColumnCounts() {
        String[][] data = {
            {"1", "2"},
            {"3"}
        };
        Class<?>[] types = {Integer.class, Integer.class};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 2));
    }

    @Test
    public void testDataToNumericMatrixPropagatesNumericConversionErrors() {
        String[][] data = {
            {"1"},
            {"not-a-number"}
        };
        Class<?>[] types = {Double.class};

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.dataToNumericMatrix(data, types, 2)
        );

        assertEquals("Invalid numeric value at row 1, column 0: not-a-number", exception.getMessage());
    }

    @Test
    public void testDataToNumericMatrixRejectsUnsupportedTypes() {
        String[][] data = {{"1"}};
        Class<?>[] types = {Object.class};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 1));
    }
}
