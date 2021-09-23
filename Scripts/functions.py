# Assemble all visitsIn columns to a vector  
assembler = VectorAssembler(
    inputCols=visitsInVars,
    outputCol="visitsInDaysVector")
df = assembler.transform(df)

# Assemble all transactionsIn columns to a vector
assembler = VectorAssembler(
    inputCols=transactionsInVars,
    outputCol="transactionsInDaysVector")
df = assembler.transform(df)

def sparse_to_array(v):
  #v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array
sparse_to_array_udf = fn.udf(sparse_to_array, types.ArrayType(types.FloatType()))
df = df.withColumn('visitsInDaysArray', sparse_to_array_udf('visitsInDaysVector'))

def sparse_to_array_2(v):
  sparse_array = list([float(x) for x in v])
  indexes = [i for i,j in enumerate(sparse_array) if j != 0]
  return indexes

sparse_to_array_udf_2 = fn.udf(sparse_to_array_2, types.ArrayType(types.IntegerType()))
df = df.withColumn('hitsCountArray', sparse_to_array_udf_2('hitsCount'))


# Concatenate all hits arrays to one
def concatenate_hits_arrays(strList):
  array_full=[]
  for i in strList: array_full = array_full+i
  return array_full

hits_array_udf = fn.udf(concatenate_hits_arrays, types.ArrayType(types.StringType()))
df = df.withColumn('hitsFullArray', hits_array_udf('hits'))

# Count Vectoriser over hits
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="hitsFullArray", outputCol="hitsCount")
model = cv.fit(df)
df = model.transform(df)

# Change the sparse vector into a python list
sparse_to_array_udf = fn.udf(sparse_to_array, types.ArrayType(types.FloatType()))
df = df.withColumn('hitsCountArray', sparse_to_array_udf('hitsCount'))

# One Hot Encoding
encoder = OneHotEncoderEstimator(inputCols=["medium_indx"], outputCols=["medium_ohe"])
df = encoder.fit(df).transform(df)
encoder = OneHotEncoderEstimator(inputCols=["deviceCategory_indx"], outputCols=["deviceCategory_ohe"])
df = encoder.fit(df).transform(df)
encoder = OneHotEncoderEstimator(inputCols=["city_indx"], outputCols=["city_ohe"])
df = encoder.fit(df).transform(df)

## padding hits
PAD_LEN = 15 
pad_fix_length = fn.udf(
    lambda arr: arr[:PAD_LEN] + [0] * (PAD_LEN - len(arr[:PAD_LEN])), 
    ArrayType(StringType())
)
df = df.withColumn('hitsPadded', pad_fix_length('hitsArray'))


# hits arrays contain string paths. convert strings to arrays of paths

def path_to_arrays(strList):
  path_full=''
  for i in strList: path_full = path_full+i
  new_array = list(filter(None, path_full.split('/'))) 
  return new_array

hits_path_to_array_udf = fn.udf(path_to_arrays, types.ArrayType(types.StringType()))
df = df.withColumn('hitsArray', hits_path_to_array_udf('pagePath'))

# Aggregating all visitsIn and transationsIn columns
aggs = [
  fn.collect_list(fn.col('visitsIn' + str(i) + 'Binary')).alias('visitsIn' + str(i)) for i in range (LOOKAHEAD+1)
]
aggs.extend([
  fn.collect_list(fn.col('transactionsIn' + str(i) + 'Binary')).alias('transactionsIn' + str(i)) for i in range (LOOKAHEAD+1)
])

df = df.groupby(
  'fullVisitorId'
).agg(
  fn.collect_list(df.visitStartTime).alias('visitStartTimes'),
  fn.collect_list(df.medium_indx).alias('channels_indx'),
  fn.collect_list(df.channelGrouping_indx).alias('channelGrouping_indx'),
  fn.collect_list(df.deviceCategory_indx).alias('devices_indx'),
  fn.collect_list(df.city_indx).alias('city_indx'),
  fn.collect_list(df.timeOnSite).alias('timeOnSite'),
  fn.collect_list(df.visitNumber).alias('visitNumber'),
  fn.collect_list(df.hitsArray).alias('hits'),
  *aggs
).orderBy(
  'fullVisitorId'
).cache()

def check_visitTime(v):
  res = all(i <= j for i, j in zip(v, v[1:])) 
  return res
check_udf = udf(check_visitTime)

df = df.withColumn('check_visitTime', check_udf('visitStartTime'))

df.select('*').where('check_visitTime = False').count()
display(df.select('*').where('check_visitTime = False'))
