from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import pandas as pd

# header=0 indicates first row has headers
data = pd.read_csv('data/maleFemaleTrain.csv')
print data.head()

target = data['Gender']
# don't want age, or else below would get all other columns
#train = data.iloc[:,1:] 
train = data.loc[:,['Weight', 'Height', 'Leg_Length', 'Arm_Length', 'Arm_Circum', 'Waist']]

print 'target dataset:'
print target.head()
print 'train dataset:'
print train.head()

# use a random forest classifier
rf = RandomForestClassifier(n_estimators=100)

# k fold cross validation
cv = cross_validation.KFold(len(train), n_folds=5, indices=False)

# run classifier on training and test cv segments
results = []
#for train, test in cv: