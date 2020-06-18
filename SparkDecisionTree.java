package Group;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import static org.apache.spark.sql.functions.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SparkDecisionTree implements Serializable {

	private String appName = "Group Spark Decision Tree";
	private JavaSparkContext context;
	private int numClasses = 2;

	public static void main(String[] args) {
		new SparkDecisionTree().run(args[0], args[1]);
	}

	public void run(String dataFile, String outputDirectory) {
		// List to store the output strings we want to write to a file
		String outputString = "";

		long start = System.nanoTime();

		SparkSession spark = SparkSession.builder()
			.appName(appName)
			.getOrCreate();
		context = new JavaSparkContext(spark.sparkContext());

		// Read and format the input dataset
		Dataset<Row> kddData = spark.read().schema(buildSchema(42)).csv(dataFile);
		Dataset<Row> indexed = new StringIndexer()
			.setInputCol("_c41")
			.setOutputCol("connection_type")
			.fit(kddData)
			.transform(kddData)
			.drop("_c41");


		// Split data 70% and 30%
		Dataset<Row>[] split = indexed.randomSplit(new double[] {0.7, 0.3});
		JavaRDD<Row> train = split[0].toJavaRDD();
		JavaRDD<Row> test  = split[1].toJavaRDD();

		JavaRDD<LabeledPoint> trainlp = getLabeledPoints(train);
		JavaRDD<LabeledPoint> testlp  = getLabeledPoints(test);

		// Generate Model
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		String impurity = "gini";
		Integer maxDepth = 3;
		Integer maxBins = 20;
		final DecisionTreeModel model = DecisionTree.trainClassifier(trainlp, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

		JavaPairRDD<Double, Double> yHatToY = testlp.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		double testErr = yHatToY.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testlp.count();
		double accuracy = 1 - testErr;

		long duration = (System.nanoTime() - start) / 1000000;  // Milliseconds

		outputString += "Test Accuracy: " + accuracy + "\n";
		outputString += "Test Error:    " + testErr + "\n";
		outputString += "Duration:      " + duration + "ms\n";
		outputString += "Learned classification tree model:\n" + model.toDebugString() + "\n";

		List<String> outputStrings = new ArrayList<String>();
		outputStrings.add(outputString);

		// Output results
		JavaRDD<String> output = context.parallelize(outputStrings);
		output.saveAsTextFile(outputDirectory);
	}

	JavaRDD<LabeledPoint> getLabeledPoints(JavaRDD<Row> set) {
		return set.map(p -> {
			int l = p.length();
			double[] features = new double[l - 2];
			for(int i = 0; i < (l - 2); i++) {
				features[i] = p.getDouble(i);
			}
			Vector v = new DenseVector(features);
			return new LabeledPoint(p.getDouble(l-1), v);
		});
	}

	StructType buildSchema(int l) {
		StructField[] f = new StructField[l];
		for(int i = 0; i < l-1; i++) {
			f[i] = DataTypes.createStructField("_c" + i, DataTypes.DoubleType, true);
		}
		f[l-1] = DataTypes.createStructField("_c" + (l-1), DataTypes.StringType, true);
		return new StructType(f);
	}
}
