import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# age: age in years (29 to 77)
# sex: (1 = male; 0 = female)
# cp: chest pain type (0 to 3)
# trestbp: sresting blood pressure (in mm Hg on admission to the hospital) (94 to 22)
# chol: serum cholestoral in mg/dl (126 to 564)
# fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# restecg: resting electrocardiographic results (0 to 2)
# thalach: maximum heart rate achieved (71 to 202)
# exang: exercise induced angina (1 = yes; 0 = no)
# oldpeak: ST depression induced by exercise relative to rest (0 to 6.2) In float
# slope: the slope of the peak exercise ST segment (0 to 2)
# ca: number of major vessels colored by flourosopy (0 to 4)
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect (3, 6, 7)
# target: 1 or 0

knn_from_joblib = joblib.load('knn_01.pkl')

list1 = [67,1,0,80,50,0,0,50,1,1.5,1,3,2]
df = pd.DataFrame([list1],columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                       'exang', 'oldpeak', 'slope', 'ca', 'thal'])
chol_max = 564
trestbps_max = 200
thalach_max = 202

df['chol'] = df['chol']/chol_max
df['trestbps']=df['trestbps']/trestbps_max
df['thalach']=df['thalach']/thalach_max

result_X = df.iloc[:, :].values
# print(result_X)

pre = knn_from_joblib.predict(result_X)
print(pre)
