# Load data
df = Utils.load_data(BUCKET, FOLDER_LABELED_DATA, DATA_FORMAT).\
    select(SELECTED_VARIABLES).cache()
print('df count:', df.count())

# Data Cleaning
# Keep only sessions for users appearing more than MIN_VISITS
df = Utils.min_visits_visitors(df, 'fullVisitorId', MIN_VISITS)
print('df count:', df.count())


# convert numeric variables to integer and fill null values
for var in NUM_VAR: df = df.withColumn(var,df[var].cast(DoubleType())).fillna(0)
  
# calculate timeDifference between visits 
partition_col = 'fullVisitorId' 
orderby_sorter_col = 'visitStartTime'

df = Utils.get_timeBetweenVisits(df, partition_col, orderby_sorter_col)

# Feature Scaling
norm_variables = ['timeOnSite', 'timeDifference']
df = Utils.normalise_cols(df, norm_variables)

# Count distinct values for the following variables
vars_count = ['medium', 'source', 'deviceCategory', 'city', 'channelGrouping']
feature_counts = Utils.get_feature_counts(df, vars_count)

# Processing and OHE hits paths
df = Utils.procces_hits(df, TOP_HITS)

# Indexing Categorical Variables
df = Utils.indexing_variables(df, CAT_VAR)

# Binary representation of columns
label_var = ['visitsIn0', 'transactionsIn0']    #label_var = visitsInVars+transactionsInVars
for var in label_var: df = df.withColumn(var, fn.when(df[var] != 0, 1).otherwise(0))
  
# Group Sessions
FEATURE_VAR = NUM_VAR + ['timeOnSite_Scaled', 'timeDifference', 'timeDifference_Scaled']
for var in CAT_VAR:FEATURE_VAR.append(var+'_index')
  
df = Utils.groupby_sort_collect(df, 'fullVisitorId', 'visitStartTime', FEATURE_VAR, ARR_VAR)
display(df) 
df.count()
## Save to parquet and continue to 03 Attr Model notebook
# DF
Utils.save_data(df, BUCKET, FOLDER_PROCESSED_DATA)
# Counts
Utils.save_df_feat_cts(df, feature_counts)
