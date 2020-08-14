package examples.classifications;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultiClassCSVTabular {

    // Training info
    private static final int numTrainData = 7000;
    private static final int epoch = 10;

    public static void main(String[] args) throws IOException, InterruptedException {

        //=====================================================================
        //            Step 1: Load & Transform data
        //=====================================================================

        RecordReader rr = loadData();

        List<List<Writable>> rawTrainData = new ArrayList<>();
        List<List<Writable>> rawTestData = new ArrayList<>();

        int idx = 0;
        while (rr.hasNext()) {
            if(idx < numTrainData) {
                rawTrainData.add(rr.next());
            } else {
                rawTestData.add(rr.next());
            }
            idx++;
        }

        System.out.println(rawTrainData.get(0));

        List<List<Writable>> transformedTrainData = transformData(rawTrainData);
        List<List<Writable>> transformedTestData = transformData(rawTestData);

        DataSetIterator trainData = makeIterator(transformedTrainData);
        DataSetIterator testData = makeIterator(transformedTestData);

//        // Double check data
//        System.out.println("\nShape of training vector:");
//        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
//        System.out.println("\nShape of test vector:");
//        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        //=====================================================================
        //            Step 2: Define Model
        //=====================================================================

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Nesterovs(0.001, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(6)
                        .nOut(20)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(7)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // System.out.println(conf.toString()); // To view model configuration in string

        //=====================================================================
        //            Step 3: Set Listener
        //=====================================================================

        model.setListeners(new ScoreIterationListener(10));

        //=====================================================================
        //            Step 4: Train model
        //=====================================================================

        Evaluation eval;
        for(int i=0; i < epoch; i++) {
            System.out.println("EPOCH: " + i);
            model.fit(trainData);
            eval = model.evaluate(testData);
            System.out.println(eval.stats());
            testData.reset();
        }

    }

    private static RecordReader loadData() throws IOException, InterruptedException {

        int numLinesToSkip = 1; // how many rows to skip. Skip header row.
        char delimiter = ',';

        // Define csv location
        File inputFile = new ClassPathResource("TabularData/AReM.csv").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // Read dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(fileSplit);

        return rr;
    }

    private static List<List<Writable>> transformData(List<List<Writable>> data) {

        //=====================================================================
        //            Define Input data schema
        //=====================================================================

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsFloat("avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23")
                .addColumnCategorical("class", Arrays.asList("walking","standing", "cycling", "sitting", "lying", "bending1", "bending2"))
                .build();

//        //Print out the schema:
//        System.out.println("Input data schema details:");
//        System.out.println(inputDataSchema);

        //=====================================================================
        //            Define transformation operations
        //=====================================================================

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .categoricalToInteger("class")
                .build();

        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();
        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);

        //=====================================================================
        //            Perform transformation
        //=====================================================================
        return LocalTransformExecutor.execute(data, tp);
    }

    private static DataSetIterator makeIterator(List<List<Writable>> data) {

        // Data info
        int labelIndex = 6; // Index of column of the labels
        int numClasses = 7; // Number of unique classes for the labels

        RecordReader collectionRecordReaderData = new CollectionRecordReader(data);
        return new RecordReaderDataSetIterator(collectionRecordReaderData, data.size(), labelIndex, numClasses);

    }
}
