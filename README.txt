### Group Project Readme ###

# Prerequisites to both applications

Firstly, there are prerequisites to being able to compile and run both applications. You'll need to copy the .java source files into
the same directory on your ECS account along with the jars directory which is obtained from:
   
   https://ecs.wgtn.ac.nz/foswiki/pub/Courses/COMP424_2020T1/Assignments/spark_jars.zip

The easiest method to obtain this file and extract is to enter the following commands while SSH'd into your ECS machine account:

   mkdir spk_grp_as3
   cd spk_grp_as3
   wget "https://ecs.wgtn.ac.nz/foswiki/pub/Courses/COMP424_2020T1/Assignments/spark_jars.zip"
   unzip spark_jars.zip
   rm spark_jars.zip

You'll also need to obtain the SetupSparkClasspath.csh file to setup the appropriate paths.

   wget "https://ecs.wgtn.ac.nz/foswiki/pub/Courses/COMP424_2020T1/Assignments/SetupSparkClasspath.csh"

And finally, while still inside the spk_grp_as3 directory, copy the files SparkDecisionTree.java and SparkLogisticRegression.java
to the directory. Your directory should look like this:

   spk_grp_as3
   ├── jars
   ├── SetupSparkClasspath.csh
   ├── SparkDecisionTree.java
   └── SparkLogisticRegression.java

The next step is to make sure you're logged into one of the cluster machines. Enter the following:

   ssh co246a-1

Afterwards, navigate to the spk_grp_as3 directory again and enter the following command to make sure your terminal is tcsh,
then source the SetupSparkClasspath.csh file:

   tcsh
   source SetupSparkClasspath.csh





# Spark Decision Tree

To compile and execute the the SparkDecisionTree.java application, you'll need to have already placed the KDD dataset file onto your
HDFS directory. For the sake of this example, we've placed the kdd.data file into the /user/username/input directory (replace username with
your actual username on the HDFS filesystem).

To compile the application, make sure to be inside the spk_grp_as3 directory and enter the following to create a jar file ready for
submission to the spark cluster:

   mkdr sdtc
   javac -cp "jars/*" -d sdtc SparkDecisionTree.java
   jar cvf sdt.jar -C sdtc .
   rm -rf sdtc

The next step is to submit the jar to the Spark cluster for exection. Enter the following:

   spark-submit --class "Group.SparkDecisionTree" \
      --master yarn \
      --deploy-mode cluster \
      sdt.jar \
      /user/username/input/kdd.data \
      /user/username/sdt_output

Note that the above assumes that the sdt_output directory does not yet exist on the HDFS system. You can change this to another name.
Once the execution is complete, you'll be able to view the output of the DecisionTree with:

   hdfs dfs -cat sdt_output/part-00001





# Spark Logistic Regression

You can follow the exact same process for compilation here as the SparkDecisionTree application, enter the following:

   mkdir slrc
   javac -cp "jars/*" -d slrc SparkLogisticRegression.java
   jar cvf slr.jar -C slrc .
   rm -rf slrc

And to execute on the Spark cluster, use the following:

   spark-submit --class "Comp424_Assessment3.SparkLogisticRegression" \
      --master yarn \
      --deploy-mode cluster \
      slr.jar \
      /user/username/input/kdd.data \
      /user/username/slr_output \
      12345676890

The final number is the seed to use for splitting the test and training sets, you can alter this to any number you'd like.
Once the run is complete, the results are stored in four different output files, some of which output intermediary results
of the classification (like predicted vs actual). View each of these files to see which contains the test and training
accuracy.

   hdfs dfs -ls slr_output

This will list the files. View each part-* file separately, replacing part-* with an actual filename obtained from the 
previous command.

   hdfs dfs -cat slr_output/part-*
