
PROCEDURES:
1.) checked for null values
print(df.isnull().any().sum())
# got zero so we are good to go

2.)check for duplicates
# check if it exist
print(df.duplicated().any())
=> True # it exist

# check how many
print(df.duplicated().sum())
=> 1 # only one duplicate exist

# check the duplicate
print(df[df.duplicated()])

# drop itn after confirming
df = df.drop_duplicates

3.) Understanding the data

# checking if the target variable is skew to we can know if we will apply target transformation
print(df['charges'].skew()) 
=> 1.515 # we are definitely using target tranformation

# plotted a scatterplot of bmi vs charges, using smoker for color dots. plotted a lineplot of age vs charges, plotted a boxplots for region and children vs charges.

# checked colleration justv in case
dep_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

df_encoded = df.copy()
df_encoded['sex'] = df_encoded['sex'].map({'male': 1, 'female': 0})
df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
df_encoded['region'] = df_encoded['region'].astype('category').cat.codes

for col in dep_columns:
    corr, p_value = stats.pearsonr(df_encoded[col], df_encoded['charges'])
    print(f"{col}: correlation={corr:.3f}, p-value={p_value:.5f}")
4.) 