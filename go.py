from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

# header=0 indicates first row has headers
data = pd.read_csv('data/worthlaylayless/maleFemaleTrain.csv')
print data.head()

# define X (features) and y (response aka target)
# feature_cols
X = data.loc[:,['Weight', 'Height', 'Leg_Length', 'Arm_Length', 'Arm_Circum', 'Waist']]
# response aka target
y = data['Gender']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)

y_results_vals = rf.predict(X_test)
y_results_probs_both = rf.predict_proba(X_test)

'''
results = []
for (a,b,c) in zip(y_test, y_results_vals, y_results_probs_both):
	results.append([a,b,c[0]])
print results
'''

# for DataFrame
y_results_probs = [ i[0] for i in y_results_probs_both]

# put it into a comprehensive pandas DataFrame
results_DF = pd.DataFrame({
	'myResultsGender': y_results_vals,
	'actualGender': y_test,
	'probGenderCorrect': y_results_probs[0]
	})

merge_DF_headers = ['Weight', 'Height', 'Leg_Length', 'Arm_Length', 'Arm_Circum', 'Waist']
merge_DF = pd.DataFrame(X_test, columns=merge_DF_headers)

# combine both DataFrames
final_DF = pd.concat([results_DF, merge_DF], axis=1)
print final_DF

# create new column, put '1' if correct, '0' if not
# can then do sum(column) / len(column) = % percentage correct
final_DF['correct'] = np.where(final_DF['actualGender']==final_DF['myResultsGender'], 1, 0)
print final_DF

# PERCENT CORRECT ------------------------------
print '----------------------------------------'

print str(( float(sum(final_DF['correct'])) / float(len(final_DF['correct'])) ) * 100) + '%'

# ----------------------------------------------
print '----------------------------------------'

# export to csv
final_DF.to_csv('final_output.csv')