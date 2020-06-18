setenv HADOOP_VERSION 2.8.0
setenv HADOOP_PREFIX /local/Hadoop/hadoop-$HADOOP_VERSION
setenv SPARK_HOME /local/spark/
setenv PATH ${PATH}:$HADOOP_PREFIX/bin:$SPARK_HOME/bin
setenv HADOOP_CONF_DIR $HADOOP_PREFIX/etc/hadoop
setenv YARN_CONF_DIR $HADOOP_PREFIX/etc/hadoop
setenv LD_LIBRARY_PATH $HADOOP_PREFIX/lib/native:$JAVA_HOME/jre/lib/amd64/server
need java8
