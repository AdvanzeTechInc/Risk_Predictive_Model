from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import train_test_split
df="./data/training/input/ClaimDetailsnew.xlsx" 

X= df.drop(columns=["E2Value"])
y = df['E2Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xgr = XGBRegressor()
xgr.fit(X_train, y_train)

from datetime import datetime
today = datetime.today().strftime('%Y%m%d')
with open('./data/training/output/xgr_'+today+'.pkl', 'wb') as file:
        pickle.dump(xgr, file)