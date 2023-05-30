# bitcamp_python_mini_project
파이썬 AI 머신러닝 회귀분석 미니프로젝트(제주도 퇴근버스 예측)





제주도 퇴근 버스 승객 예측. 머신러닝 프로젝트 코드리뷰




데이터를 살펴보는 과정.



x data는 오전부터 점심시간 까지의 탑승 승객 수와 버스에 대한 정보 및 날짜정보이다.

y data는 18~20ride 이다.



아래 이미지를 보면 유의미한 상관관계를 나타내는 부분을 확인해볼 수 있다.


#1. 데이터 전처리



# 1. Data preprocessing
# Read the dataset from a CSV file
path = './'
datasets = pd.read_csv(path + 'train.csv')

# Extract the relevant features from the dataset
x = datasets[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
              'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
              '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
              '8~9_takeoff', '9~10_takeoff','10~11_takeoff','11~12_takeoff']].copy()

# Create additional features based on ride and takeoff categories
x['takeon_avg_6~8'] = (x['6~7_ride'] + x['7~8_ride']) / 2
x['takeon_avg_8~10'] = (x['8~9_ride'] + x['9~10_ride']) / 2
x['takeon_avg_10~12'] = (x['10~11_ride'] + x['11~12_ride']) / 2
x['takeon_avg_ride'] = (x['takeon_avg_6~8'] + x['takeon_avg_8~10'] + x['takeon_avg_10~12']) / 3

x['takeoff_avg_6~8'] = (x['6~7_takeoff'] + x['7~8_takeoff']) / 2
x['takeoff_avg_8~10'] = (x['8~9_takeoff'] + x['9~10_takeoff']) / 2
x['takeoff_avg_10~12'] = (x['10~11_takeoff'] + x['11~12_takeoff']) / 2

# Convert the 'date' column to datetime format
x['date'] = pd.to_datetime(datasets['date']) 

# Extract additional features from the 'date' column
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x['weekday'] = x['date'].dt.weekday

# Drop the 'date' column from the feature set
x = x.drop('date', axis=1)

# Calculate additional statistical aggregations
ride_columns = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']
takeoff_columns = ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff','11~12_takeoff']

x['takeoff_median'] = x[takeoff_columns].median(axis=1)
x['ride_max'] = x[ride_columns].max(axis=1)
x['ride_min'] = x[ride_columns].min(axis=1)

# Categorize weekdays and weekends
x['is_weekend'] = np.where(x['weekday'] < 5, 0, 1)

# Convert 'in_out' feature to numerical values using mapping
x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})

# Map unique station names to numerical values
station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
x['station_name'] = x['station_name'].map(station_name_mapping)
x_encoded = pd.get_dummies(x, columns=['station_name'])
x_encoded = x_encoded.fillna(0)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, train_size=0.85, random_state=80, shuffle=True
)
이번  프로젝트는 독립변수 x 를 추가할 수 없다는 조건 하에서 진행하게되었다.
따라서 나는 제한적인 상황에서 정확도를 높이기 위해 x 데이터를 추가하기 위해서
아래와 같은 전처리를 하였다.
(* 결측치는 존재하지않았고 이상치 제거는 실제 승객 숫자를 나타내는 정보이기 때문에 ,
예측 모델을 학습시킬 때 이상치가 유용한 정보를 포함하고 있을 수 있다고 판단하였다.)

1. 퇴근 시간 버스 승객 예측에 상관관계가 부족하다고 생각되어 takeoff 11~12 를 제거
2. 1시간 간격으로 count 되었던 승객수를 2시간 간격의 범주로 추가
3. 평균값,중앙값,최대값,최솟값 등을 추가
4. date를 연/월/일/요일로 구분하여 컬럼을 추가하고 기존 date는 삭제
5. 요일 데이터를 평일과 주말로 다시 구분
6. 버스ID를 시내와 시외로 구분해주기 위해 맵핑
7. 원핫인코딩을 통해서 station name 을 맵핑하여 숫자를 부여 (다차원의 벡터로 변환)

일련의 과정들을 통해서 데이터 수를 늘리고 정확도를 높이며 
mse 값을 낮출 수 있었다.

이 과정에서 사실은 히트맵을 그려보고 데이터 분포를 확인함을 통해서
그럴 듯한 논리로 진행하였기는 하지만, 사실은 실제로 분석을 진행해보면 의외로 정확도가
내려가는 경우가 많았다.

때문에 주석 처리를 해두고 삭제해나아가는 과정을 통해서  최적의 조합값을 찾는 수 밖에 없었다.
더 배울 기회가 생긴다면 어떠한 이유로, 또 어떠한 정제가 영향을 더 크게 미치는지에 대한 연구를 해보고 싶다.
#. 하이퍼파라미터 설정
# Define the hyperparameters for the BaggingRegressor
param = {
    'n_estimators': [3947],
    'depth': [16],
    'fold_permutation_block': [237],
    'learning_rate': [0.8989964556692867],
    'od_pval': [0.6429734179569129],
    'l2_leaf_reg': [2.169943087966259],
    'random_state': [1417]
}

# Define the BaggingRegressor model with base estimator as DecisionTreeRegressor
bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    max_features=7,
    n_estimators=100,
    n_jobs=-1,
    random_state=62
)

# Define the cross-validation strategy using KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform grid search with cross-validation to find the best hyperparameters
model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)

# Extract the best hyperparameters from the grid search results
depth = param['depth'][0]
l2_leaf_reg = param['l2_leaf_reg'][0]
border_count = param['fold_permutation_block'][0]

print(f"depth: {depth}")
print(f"l2_leaf_reg: {l2_leaf_reg}")
print(f"border_count: {border_count}")

# Scale the features using MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
Optuna 로 최적 파라미터를 찾아 적용해주었지만, 오히려 파라미터를 적용하면 정확도가 낮아지는 현상이
나타났다. 때문에 배깅과 파라미터를 적용한 model로 fit 하지는 않았다.

가장 예측성이 높았던 시도는 catboost와 xgboost를 동시에 적용시킨 Ensemble 모델이었다.
단, 그 차이가 크지 않으며 또한 mse 값이 매우 높게 나타났기 때문에
boost 3계열을 voting 하여 시도한 것 중 가장 결과가 안정적인 Catboost를 채택했다.

이외에 Select model , Feature Importances 등을 수행하였다.
하지만 Default Param 의 경우에 가장 좋은 성능을 보였다.

궁금하여 공식문서를 포함한 여러 글들 통해서 일반적으로 데이터가 선형 관계에 가까울수록,
데이터가 균일하게 분포되어 있을수록, 변수들이 독립적일수록, 데이터의 크기가 크다면
모델의 기본 파라미터가 좋은 결과를 보인다고 해석할 수 있었다.
#2-3. 모델 설정 및 예측평가 (voting)
# 2. Model construction
# Import the necessary models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#3. training and evaluation prediction
# Initialize the models
xgb = XGBRegressor()
cat = CatBoostRegressor()
lgbm = LGBMRegressor()

model = VotingRegressor(
    estimators=[('xgb', xgb), ('lgbm',lgbm), ('cat', cat)], 
    n_jobs=-1
)

for model in regressors:
    # Train the model
    model.fit(x_train, y_train)
    
    # Make predictions on the test set
    y_predict = model.predict(x_test)
    
    # Evaluate the model using R-squared score
    score = r2_score(y_test, y_predict)
    
    # Print the model's accuracy
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4}'.format(class_names, score))
voting 을 통해서 boost 3계열의 정확도를 같이 평가하였다.
boost 3계열만을 선택하여 정확도를 평가한 이유는 다음과 같다.

우선 All_Estimator 를 통해서 여러 모델의 정확도를 평가해본 결과,
boost 3계열이 가장 높으며 서로 유사한 정확도를 보였기 때문이다.
이후 데이터 변동에 따른 서로의 변화를 보기 위해 voting 을 사용하였다.
#4. 결과 시각화
#4. visualization
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_predict)
plt.plot(y_test, y_predict, color='Red')
plt.show()

# Scatter plot of actual vs predicted values with a line of perfect prediction
plt.figure(figsize=(8, 6))
plt.scatter(y_test.values.ravel(), y_predict)
plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.show()

# Heatmap of correlation coefficients
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.set(rc={'figure.figsize':(20, 15)})
sns.heatmap(data=datasets.corr(),
            square=True,
            annot=True,
            cbar=True,
            cmap='coolwarm'
            )
plt.show()










나는 팀프로젝트에서 모델링과 파라미터, 알고리즘 분석을 맡았다.

독립변수를 추가할 수 없다는 프로젝트의 규정상 정확도를 개선하기위해서 가능한 방법을 전부 시도했었던 것 같다.



Ensemble, KFold, Feature Importances, GridSearchCV, Bagging, Voting

Optuna, QuantileTransformer, SelectFromModel



scaler 및 model 등은 데이터가 바뀔 때마다 새로 적용시켜서

가장 정확도가 높은 값을 찾기 위해서 노력했던 것 같다.



다음 프로젝트에는 적용시킨 모델,파라미터 등 값에 변화에 따른 정보를 기록하고

이를 저장하여 시각화하는 것도 아주 유용할 것 같다는 생각이 들었다.
