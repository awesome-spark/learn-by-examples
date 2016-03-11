This document is a guide to achieve scoring with Spark through a practical case. Inspired from a R tutorial [here](http://rstudio-pubs-static.s3.amazonaws.com/5267_0156db47a0604aa9818143e7d2db226e.html).
It is not on the mathematical aspects of the construction of a score but rather encourages the use of Spark and exploitation of results. It is divided into four parts:
<br>

1. Objective description of the study and data
2. Data preparation and initial analysis
3. Construction and validation of the score
4. Interpretation of results

## Objective of the study and description of the data

### Introduction

Scoring is a technique for prioritizing data for assessing a rating or score the probability that an individual meets a solicitation or belong to the intended target.

The score is usually obtained from the quantitative and qualitative data available on the individual (socio-demo data, purchasing behavior, previous answers ...) to which are applied a scoring model.

In general, the modeling technique used is the logistic regression. it is one of supervised learning techniques, i.e that one wishes to explain in general belonging to a category from descriptors collected on a population sample in order to generalize learning.
<br>

Some examples of applications:

- Determine the viability of a client seeking a credit from its characteristics (age, type of job, income level, other outstanding loans, etc.)
- For a company determine the best type of planting area based on neighborhood characteristics (SPC, Number of inhabitants, Life Cycle, etc.)

### Prepare Raw Data

Let's first load the needed libraries. We are working with Zeppelin Notebook (v.0.6.0) :

```scala
%dep

z.load("com.databricks:spark-csv_2.11:1.3.0")
```

We'll need to read the data raw first using `spark-csv` :

```scala
val dataPath = "/home/eliasah/Desktop/r-snippets/heart-disease-study/data/SAheart.data.txt"
val rawData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(dataPath).drop("row.names")
rawData.show

// +---+-------+-----+---------+-------+-----+-------+-------+---+---+
// |sbp|tobacco|  ldl|adiposity|famhist|typea|obesity|alcohol|age|chd|
// +---+-------+-----+---------+-------+-----+-------+-------+---+---+
// |160|   12.0| 5.73|    23.11|Present|   49|   25.3|   97.2| 52|  1|
// |144|   0.01| 4.41|    28.61| Absent|   55|  28.87|   2.06| 63|  1|
// |118|   0.08| 3.48|    32.28|Present|   52|  29.14|   3.81| 46|  0|
// |170|    7.5| 6.41|    38.03|Present|   51|  31.99|  24.26| 58|  1|
// |134|   13.6|  3.5|    27.78|Present|   60|  25.99|  57.34| 49|  1|
// |132|    6.2| 6.47|    36.21|Present|   62|  30.77|  14.14| 45|  0|
// |142|   4.05| 3.38|     16.2| Absent|   59|  20.81|   2.62| 38|  0|
// |114|   4.08| 4.59|     14.6|Present|   62|  23.11|   6.72| 58|  1|
// |114|    0.0| 3.83|     19.4|Present|   49|  24.86|   2.49| 29|  0|
// |132|    0.0|  5.8|    30.96|Present|   69|  30.11|    0.0| 53|  1|
// |206|    6.0| 2.95|    32.27| Absent|   72|  26.81|  56.06| 60|  1|
// |134|   14.1| 4.44|    22.39|Present|   65|  23.09|    0.0| 40|  1|
// |118|    0.0| 1.88|    10.05| Absent|   59|  21.57|    0.0| 17|  0|
// |132|    0.0| 1.87|    17.21| Absent|   49|  23.63|   0.97| 15|  0|
// |112|   9.65| 2.29|     17.2|Present|   54|  23.53|   0.68| 53|  0|
// |117|   1.53| 2.44|    28.95|Present|   35|  25.89|  30.03| 46|  0|
// |120|    7.5|15.33|     22.0| Absent|   60|  25.31|  34.49| 49|  0|
// |146|   10.5| 8.29|    35.36|Present|   78|  32.73|  13.89| 53|  1|
// |158|    2.6| 7.46|    34.07|Present|   61|   29.3|  53.28| 62|  1|
// |124|   14.0| 6.23|    35.96|Present|   45|  30.09|    0.0| 59|  1|
// +---+-------+-----+---------+-------+-----+-------+-------+---+---+

rawData.describe().show()

// +-------+------------------+-----------------+-----------------+------------------+------------------+------------------+------------------+------------------+------------------+
// |summary|               sbp|          tobacco|              ldl|         adiposity|             typea|           obesity|           alcohol|               age|               chd|
// +-------+------------------+-----------------+-----------------+------------------+------------------+------------------+------------------+------------------+------------------+
// |  count|               462|              462|              462|               462|               462|               462|               462|               462|               462|
// |   mean|138.32683982683983| 3.63564935064935|4.740324675324673|25.406731601731614|53.103896103896105|26.044112554112548|17.044393939393945|42.816017316017316|0.3463203463203463|
// | stddev|20.496317175467627|4.593024078404592|2.070909161059325| 7.780698595839762| 9.817534115584072| 4.213680226897767|24.481058691658575|14.608956444552494|0.4763125365907826|
// |    min|               101|              0.0|             0.98|              6.74|                13|              14.7|               0.0|                15|                 0|
// |    max|               218|             31.2|            15.33|             42.49|                78|             46.58|            147.19|                64|                 1|
// +-------+------------------+-----------------+-----------------+------------------+------------------+------------------+------------------+------------------+------------------+
```

We will notice that we have a categorical feature `famhist` for family history. Thus, we'll be needed to encode it for further usage. We will also need to convert the `chd` feature into a `DoubleType`:

```scala
// famhist UDF encoder
val encodeFamHist = udf[Double, String]{ _ match { case "Absent" => 0.0 case "Present" => 1.0} }

// Apply UDF and cast on data
val data = rawData.withColumn("famhist",encodeFamHist('famhist)).withColumn("chd",'chd.cast("Double"))
```
## Categorical feature encoder

```scala
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val toVec = udf[Vector, Double] { (a) =>  Vectors.dense(a) }

import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder().setInputCol("chd").setOutputCol("chd_categorical")

val encoded = encoder.transform(data).toDF
encoded.registerTempTable("encoded")
```
