package Comp424_Assessment3;

import jodd.typeconverter.Convert;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.ChiSqSelectorModel;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.LinkedList;
import java.util.List;

public class SparkLogisticRegression {

    public static void main(String[] args) {
        // TODO: Following code block is for local only - remove this on the hadoop cluster.
        //System.setProperty("hadoop.home.dir","D:\\VUW\\COMP424\\COMP424Hadoop\\Hadoop");
        //args = new String[3];
        //args[0] = "DataToProcess\\kdd_input";
        //args[1] = "DataToProcess\\kdd_output";
        //args[2] = "333";

        long startTime = System.nanoTime();

        // Save our results as RDD records so we can output them in Hadoop.
        List<Row> lrModelResults = new LinkedList<Row>();
        StructType lrResultsSchema = new StructType(new StructField[]{
            new StructField("Result Description", DataTypes.StringType, false, Metadata.empty()),
            new StructField("Result Value", DataTypes.StringType, false, Metadata.empty())
        });

        lrModelResults.add(RowFactory.create("Random Seed", args[2]));

        // Configure Spark session.
        SparkSession spark = SparkSession.builder()
                .appName("SparkLogisticRegression")
                //.master("local") // TODO: When working locally - remove this on the hadoop cluster.
                .getOrCreate();

        // Read data as CSV file to process.
        JavaRDD<String> lines = spark.read().textFile(args[0]).toJavaRDD();

        JavaRDD<LabeledPoint> linesRDD = lines.map(line -> {
            String[] tokens = line.split(","); // Comma separated
            double[] features = new double[tokens.length - 1];

            for (int i = 0; i < features.length; i++) {
                features[i] = Double.parseDouble(tokens[i]);
            }

            Vector vectorFeatures = new DenseVector(features);

            if (tokens[features.length].equals("normal")) { // normal class label
                return new LabeledPoint(0.0, vectorFeatures);
            }
            else { // anomaly class label
                return new LabeledPoint(1.0, vectorFeatures);
            }
        });

        Dataset<Row> kddData = spark.createDataFrame(linesRDD, LabeledPoint.class);

        // Create training and test sets with random 70/30 split.
        Dataset<Row>[] randomSplit = kddData.randomSplit(new double[]{0.7, 0.3}, Convert.toInteger(args[2]));
        Dataset<Row> trainingData = randomSplit[0];
        Dataset<Row> testData = randomSplit[1];

        // Get the count of records as sanity check and to check the class label balance.
        double trainingDataRows = trainingData.count();
        double testDataRows = testData.count();

        lrModelResults.add(RowFactory.create("Training set record count", Double.toString(trainingDataRows)));
        lrModelResults.add(RowFactory.create("Test set record count", Double.toString(testDataRows)));
        //System.out.println("Records in training set: " + trainingData.count());
        //System.out.println("Records in test set: " + testData.count());

        // The set is pretty well balanced, so shouldn't have to deal with any imbalance.
        double normalCount = trainingData.select("label").where("label = 0").count();
        double anomalyCount = trainingData.select("label").where("label = 1").count();

        lrModelResults.add(RowFactory.create("Training data NORMAL label count", Double.toString(normalCount)));
        lrModelResults.add(RowFactory.create("Training data NORMAL label %", Double.toString(normalCount / trainingDataRows * 100)));
        lrModelResults.add(RowFactory.create("Training data ANOMALY label count", Double.toString(anomalyCount)));
        lrModelResults.add(RowFactory.create("Training data ANOMALY label %", Double.toString(anomalyCount / trainingDataRows * 100)));
        //System.out.println("The number of normal labels = " +  numNormal);
        //System.out.println("Percentage of normal labels = " + numNormal / trainingDataSize * 100);

        // Feature selection. Try Chi-squared, which seems to be common for binomial labeled datasets.
        ChiSqSelector featureSelector = new ChiSqSelector()
                //.setSelectorType("fpr")
                //.setFpr(0.05)
                .setNumTopFeatures(30) // Top 30 seems to reap similar results as with all 41.
                .setFeaturesCol("features")
                .setLabelCol("label")
                .setOutputCol("selectedFeatures");

        ChiSqSelectorModel featureSelectorModel = featureSelector.fit(trainingData);

        trainingData = featureSelectorModel.transform(trainingData);
        testData = featureSelectorModel.transform(testData);

        // Define the Logistic Regression instance
        LogisticRegression logisticRegression = new LogisticRegression()
            .setFeaturesCol("selectedFeatures") // Change this to "selectedFeatures" to use Chisq features selection.
            .setLabelCol("label")
            .setMaxIter(10) // Set maximum iterations.
            .setRegParam(0.3) // Set Lambda.
            .setElasticNetParam(0.8); // Set Alpha.

        // Fit the model.
        LogisticRegressionModel logisticRegressionModel = logisticRegression.fit(trainingData);

        lrModelResults.add(RowFactory.create("Model coefficients", logisticRegressionModel.coefficients().toString()));
        lrModelResults.add(RowFactory.create("Model intercept", Double.toString(logisticRegressionModel.intercept())));
        //System.out.println("Coefficients: " + logisticRegressionModel.coefficients() + " Intercept: " + logisticRegressionModel.intercept());

        // Extract the summary from the returned model.
        BinaryLogisticRegressionTrainingSummary trainingSummary = logisticRegressionModel.binarySummary();

        // Obtain the loss per iteration.
        //double[] objectiveHistory = trainingSummary.objectiveHistory();

        //for (double lossPerIteration : objectiveHistory) {
        //    System.out.println(lossPerIteration);
        //}

        // Obtain the ROC as a data frame and areaUnderROC.
        Dataset<Row> roc = trainingSummary.roc();
        roc.repartition(1).write().mode("overwrite").format("csv").option("header", "true").save(args[1]); // Could graph the ROC curve with this...
        lrModelResults.add(RowFactory.create("Training AUC", Double.toString(trainingSummary.areaUnderROC())));
        //roc.show(); // False positive and true positive rates for model.
        //System.out.println(trainingSummary.areaUnderROC()); // AUC measurement for performance - technically higher be better, ranging from 0 - 1.

        // Get the threshold corresponding to the maximum F-Measure.
        Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();

        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                                    .select("threshold")
                                    .head()
                                    .getDouble(0);

        // Set the best (maximum) selected threshold for the model.
        lrModelResults.add(RowFactory.create("Best model threshold", Double.toString(bestThreshold)));
        //System.out.println("Best threshold = " + bestThreshold);
        logisticRegressionModel.setThreshold(bestThreshold);

        // Make predictions on the unseen test set.
        Dataset<Row> lrModelPredictionsTrain = logisticRegressionModel.transform(trainingData);
        Dataset<Row> lrModelPredictionsTest = logisticRegressionModel.transform(testData);

        // Select example rows to display.
        //lrModelPredictionsTest.show(10);

        lrModelPredictionsTrain.select("label", "prediction").repartition(1).write().mode("append").format("csv").option("header", "true").save(args[1]);
        lrModelPredictionsTest.select("label", "prediction").repartition(1).write().mode("append").format("csv").option("header", "true").save(args[1]);

        // This is in the example, but the Binary one seems more suitable, surely.
        // Select (prediction, true label) and compute test error.
        /*MulticlassClassificationEvaluator lrEvaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");

        double lrModelAccuracy = lrEvaluator.evaluate(lrModelPredictions);
        System.out.println("Test error larger is better = " + lrEvaluator.isLargerBetter());
        System.out.println("Test Error = " + (1.0 - lrModelAccuracy)); */

        // Select (prediction, true label) and compute test error.
        BinaryClassificationEvaluator lrEvaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction");

        double lrModelAccuracyTrain = lrEvaluator.evaluate(lrModelPredictionsTrain);
        double lrModelAccuracyTest = lrEvaluator.evaluate(lrModelPredictionsTest);

        lrModelResults.add(RowFactory.create("Model test Error (AOC) for training data", Double.toString(lrModelAccuracyTrain)));
        lrModelResults.add(RowFactory.create("Model test Error (AOC) for test data", Double.toString(lrModelAccuracyTest)));

        //System.out.println("Test error larger is better = " + lrEvaluator2.isLargerBetter());
        //System.out.println("Model test Error (AOC) for training data = " + lrModelAccuracyTrain);
        //System.out.println("Model test Error (AOC) for test data = " + lrModelAccuracyTest);

        // TODO: Does cross validtion for tuning parameters make a differnce from the default values?

        // Make grids for optimal hyper parameter selection via cross validation.
        ParamMap[] paramGrid = new ParamGridBuilder()
                //.addGrid(logisticRegression.aggregationDepth(), new int[] {2, 5, 10})
                .addGrid(logisticRegression.elasticNetParam(), new double[] {0.0, 0.5, 1.0})
                .addGrid(logisticRegression.fitIntercept())
                .addGrid(logisticRegression.maxIter(), new int[] {10, 100, 250})
                .addGrid(logisticRegression.regParam(), new double[] {0.01, 0.5, 2.0})
                .build();

        CrossValidator cv = new CrossValidator()
                .setEstimator(logisticRegression)
                .setEvaluator(lrEvaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)  // Use 3+ in practice
                .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel

        CrossValidatorModel cvModel = cv.fit(trainingData);

        lrModelResults.add(RowFactory.create("CV tuned model parameter 'elasticNet'", cvModel.bestModel().extractParamMap().apply(cvModel.bestModel().getParam("elasticNetParam")).toString()));
        lrModelResults.add(RowFactory.create("CV tuned model parameter 'fitIntercept'", cvModel.bestModel().extractParamMap().apply(cvModel.bestModel().getParam("fitIntercept")).toString()));
        lrModelResults.add(RowFactory.create("CV tuned model parameter 'maxIter'", cvModel.bestModel().extractParamMap().apply(cvModel.bestModel().getParam("maxIter")).toString()));
        lrModelResults.add(RowFactory.create("CV tuned model parameter 'regParam'", cvModel.bestModel().extractParamMap().apply(cvModel.bestModel().getParam("regParam")).toString()));

        Dataset<Row> cvModelPredictionsTrain = cvModel.transform(trainingData);
        Dataset<Row> cvModelPredictionsTest = cvModel.transform(testData);

        double cvModelAccuracyTrain = lrEvaluator.evaluate(cvModelPredictionsTrain);
        double cvModelAccuracyTest = lrEvaluator.evaluate(cvModelPredictionsTest);

        lrModelResults.add(RowFactory.create("CV tuned model test accuracy (AOC) for training data", Double.toString(cvModelAccuracyTrain)));
        lrModelResults.add(RowFactory.create("CV tuned model test accuracy (AOC) for test data", Double.toString(cvModelAccuracyTest)));

        long endTime = System.nanoTime();
        long duration = (endTime - startTime) / 1000000;  // Milliseconds
        lrModelResults.add(RowFactory.create("Model running time", Long.toString(duration)));

        // Output results to specified Hadoop output directory.
        Dataset<Row> lrResultsFile = spark.createDataFrame(lrModelResults, lrResultsSchema);
        lrResultsFile.repartition(1).write().mode("append").format("csv").option("header", "true").save(args[1]);
    }
}
