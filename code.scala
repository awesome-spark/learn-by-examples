val dataPath = "/media/eliasah/Transcend/bitbucket/scoring-heart-disease/data/SAheart.data.txt"
val rawData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(dataPath).drop("row.names")

// 
z.show(rawData)
