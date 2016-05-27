# Scoring Heart Diseases with Apache Spark

This document is a guide to achieve scoring with Apache Spark through a practical case. Inspired from a R tutorial [here](http://rstudio-pubs-static.s3.amazonaws.com/5267_0156db47a0604aa9818143e7d2db226e.html).
It is not on the mathematical aspects of the construction of a score but rather encourages the use of Spark and exploitation of results. It is divided into four parts:

1. Objective description of the study and data
2. Data preparation and initial analysis
3. Construction and validation of the score
4. Interpretation of results

## Objective of the study and description of the data

### Introduction

Scoring is a technique for prioritizing data for assessing a rating or score the probability that an individual meets a solicitation or belong to the intended target.

The score is usually obtained from the quantitative and qualitative data available on the individual (socio-demo data, purchasing behavior, previous answers ...) to which are applied a scoring model.

In general, the modeling technique used is the logistic regression. it is one of supervised learning techniques, i.e that one wishes to explain in general belonging to a category from descriptors collected on a population sample in order to generalize learning.

Some examples of applications:

- Determine the viability of a client seeking a credit from its characteristics (age, type of job, income level, other outstanding loans, etc.)
- For a company, determine the best type of planting area based on neighborhood characteristics (SPC, Number of inhabitants, Life Cycle, etc.)

### Case Study

We are interested in our case study in a database containing data about 462 patients for whom we want to predict exposure to a heart attack.

The data is available [here](http://statweb.stanford.edu/~tibs/ElemStatLearn/) under the tab Data > South African Heart Disease.

#### Data Description

A retrospective sample of males in a heart-disease high-risk region of the Western Cape, South Africa. There are roughly two controls per case of CHD. Many of the CHD positive men have undergone blood pressure reduction treatment and other programs to reduce their risk
factors after their CHD event. In some cases the measurements were made after these treatments. These data are taken from a larger dataset, described in  Rousseauw et al, 1983, South African Medical Journal.

| variable | description
|-----| ----- |
| sbp	|	systolic blood pressure |
| tobacco	|	cumulative tobacco (kg) |
| ldl |	low densiity lipoprotein cholesterol |
| adiposity | |
| famhist	|	family history of heart disease (Present, Absent) |
| typea	|	type-A behavior |
| obesity | |
| alcohol	|	current alcohol consumption |
| age	|	age at onset |
| chd	|	response, coronary heart disease |

For our study case, we will be interested in that last variable `chd`.

### Model Definition

We aim to determine the probability of a given heart disease observations on 462 patients such as chd = f (obesity, age, family history, etc ...)

> ***Yes, that not big data, why do we need spark for that ?*** *We can use R or Pandas.*

> *Well the whole point is actually to define this analytical gait on Spark and Zeppelin which can be also carried to use over big data.*

So back to our example ! This example illustrates  how to use logistic regression on health data. The problem formulation is generally the same, it can be for an insurance company to determine the risk factors to provision (pricing) determine the profiles of palatable customers a new commercial offer, and also in macroeconomics, this approach is used to quantify country risk.

#### Modeling approach

Like any good modeling approach, building a good scoring model is a succession of more or less basic steps based practitioners. Nevertheless they all agree more or less to respect the following:

- Exploratory Analysis: What is the data set? Are there any missing values?
- Check the correlation between the descriptors and the variable.
- Identify important and redundant predictors to create a parsimonious model (this step is very important when you want to make forecasts).
- Estimate the model on a training sample.
- Validate the model on a test sample and build the model based on quality indicators.
- Compare different models and retaining the most suitable model according to the purpose of the study.

We will follow these steps and solve the problem of our case study.


### Modelization

#### 1. Load Raw Data

Let's first load the needed libraries. We are working with Zeppelin Notebook (v.0.6.0). We will need the `spark-csv` package to read the downloaded data into a Spark `DataFrame`.

In Zeppelin, we edit the first cell adding the following lines to load the dependency.

```scala
%dep

z.load("com.databricks:spark-csv_2.11:1.3.0")
```

In a separate cell, we'll need to read the data raw first using `spark-csv` as followed :

```scala
// I believe that you are old enough to know where you've put your downloaded
// data and thus change to the according path
val dataPath = "./heart-disease-study/data/SAheart.data.txt"
val rawData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(dataPath).drop("row.names")
```

We can now check what our data looks like. Let's check the content of the data.

```scala

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
```

or with `z.show(rawData)` :

![alt text](/media/eliasah/Transcend/bitbucket/scoring-heart-disease/figures/ZShowRawData.png)

#### 2. Exploratory Analysis

Hence the data is already loaded, we can check the type of each columns by printing the schema :

```scala
rawData.printSchema
// root
//  |-- sbp: integer (nullable = true)
//  |-- tobacco: double (nullable = true)
//  |-- ldl: double (nullable = true)
//  |-- adiposity: double (nullable = true)
//  |-- famhist: string (nullable = true)
//  |-- typea: integer (nullable = true)
//  |-- obesity: double (nullable = true)
//  |-- alcohol: double (nullable = true)
//  |-- age: integer (nullable = true)
//  |-- chd: integer (nullable = true)
```

Note that the chd target variable was treated as a numeric variable. We will deal with that later on.

Let's run some summary statistics on the data (e.g `z.show(rawData.describe())`):

![alt text](/media/eliasah/Transcend/bitbucket/scoring-heart-disease/figures/ZShowRawDataSummary.png)

#### Categorical feature encoder

We will also notice that we have a categorical feature `famhist` for family history. Thus, we'll need to encode it for further usage. We will also need to convert the `chd` feature into a `DoubleType` before converting it into a categorical feature:

```scala
// famhist UDF encoder
val encodeFamHist = udf[Double, String]{
  _ match { case "Absent" => 0.0 case "Present" => 1.0}
}

// Apply UDF and cast on data
val data = rawData
              .withColumn("famhist",encodeFamHist('famhist))
              .withColumn("chd",'chd.cast("Double"))
```

```scala
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.Pipeline

val toVec = udf[Vector, Double] { (a) =>  Vectors.dense(a) }

val encodeFamHist = udf[Double, String]( _ match { case "Absent" => 0.0 case "Present" => 1.0} )      
val data = base.withColumn("famhist",encodeFamHist('famhist)).withColumn("chd",'chd.cast("Double"))

val chdEncoder = new OneHotEncoder().setInputCol("chd").setOutputCol("chd_categorical")
val famhistEncoder = new OneHotEncoder().setInputCol("famhist").setOutputCol("famhist_categorical")

val pipeline = new Pipeline().setStages(Array(chdEncoder, famhistEncoder))

val encoded = pipeline.fit(data).transform(data)
```

#### 3. Search meaningful explanatory variables

Attention when conducting a graph analysis is for the purpose of detecting possible colinearities, or at least to have some ideas. The variables to consumption of alcohol and the quantity of tobacco seem to be distributed in the same way, as well as cholesterol and obesity.

Another analysis tool is to perform point cloud for all variables. One can possibly color the points according to the target variable.

#### 4. Outliers and missing values

Outliers depend on the distribution and most certainly the object of the study. In the literature, the treatment of missing values or outliers is sometimes subject to endless discussions which practitioners should consider. It is easier, in general, to decide what is an outlier with some domain-knowledge. Let's look again basic statistics

The distribution of tobacco consumption is very spread out, as for alcohol. Other distributions seem rather consistent. So, for now, we do nothing on those values considered, a priori, as absurd given the distribution.

#### 5. Discretize or not?

This is a common issue in the exploratory analysis. Discretizing continuous variables.

In the data set, the variables:

- age
- tobacco
- sbp
- alcohol
- obesity
- ldl
- adiposity
- typea

are potentially discrétiser.

Should we discretize continuous variables? Yes, mostly. But how? in line with the target variable? For business knowledge? Distribution based on quantiles?

No definitive answer. From a general point of view the choice of method generally depends on the problem, the time you want to spend. Always remember: No cutting is good a priori and based on practical results, do not hesitate to reconsider its cutting.

The variable age is the simplest generally to discretize. Heart problems do not affect in the same way according to age, as shown [here](https://en.wikipedia.org/wiki/Heart_failure)
We made the choice to discretize the variables age and tobacco distinguishing between small, medium and heavy smoker.


The category under 15 years is not at all representative in the sample. the under 25 either.
Half of people over 45 are suffering from a heart problem.
One can think of a form of hereditary heart problems considering genetic's crossing.
It is clear that smoking has a real influence on heart problems, because we found a significant proportion of people,regardless of their amount of tobacco consumed.

These initial analyzes indicate that:

- It is not very useful to keep the sample in individuals under 15 years, because the model we develop is not calibrated to predict the likelihood of developing heart disease if age plays a role therefore.
- Some results that we see here descriptively must be able to confirm in a modeling phase. Our baseline will be:

```r
base3 <- subset(x = base2, subset = (age > 15))
detach(base2)
attach(base3)
## The following object(s) are masked _by_ '.GlobalEnv':
##
##     age.d, tobacco.d
## The following object(s) are masked from 'base':
##
##     adiposity, age, alcohol, chd, famhist, ldl, obesity, sbp,
##     tobacco, typea
# On enlève dans la base de travail les variables age et tobacco
# continues, au profit des discrétisés
base3 <- subset(base3, select = -c(age, tobacco))
```

#### 6. Sampling : Training vs test
#### 7. Build models
#### 8. Modelization
#### 9. Model validation
#### 10. ROC
