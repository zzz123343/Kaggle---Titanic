# Kaggle - Titanic 

## 1.載入所需套件與數據:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

train = pd.read_csv(r"C:\Users\zzz12\Desktop\data\titanic\train.csv", encoding="utf-8")
test = pd.read_csv(r"C:\Users\zzz12\Desktop\data\titanic\test.csv", encoding="utf-8")
```

## 2. 確認缺失值和資料屬性:

```python
# 確認缺失值和查看
train.info()
test.info()
```
### train.info()  
![](https://github.com/zzz123343/Kaggle---Titanic-/raw/main/images/0-1.png)  
### test.info()  
![](https://github.com/zzz123343/Kaggle---Titanic-/raw/main/images/0-2.png)  

```python
#查看各項數據
train.describe()
test.describe()
```
### train.describe()  
![](https://github.com/zzz123343/Kaggle---Titanic-/raw/main/images/1-1.png)  
###test.describe()  
![](https://github.com/zzz123343/Kaggle---Titanic-/raw/main/images/1-2.png)  

## 3. 處理缺失值:
```python
# 用年齡的中位數填補缺失的年齡值
train['Age'].fillna(train['Age'].median(), inplace=True)
# 用眾數填補缺失的登船點
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
# 用艙等的平均票價填補缺失的票價
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)

# 同樣處理 test 數據集中的缺失值
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)
```

## 4. 特徵工程:
```python
# 為了進一步分析，提取乘客的稱謂（例如 Mr, Mrs 等），這可能與生存率有關
for df in [train, test]:
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Dona', 'Jonkheer'], 'Rare')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
    df['Title'] = df['Title'].replace('Don', 'Mr')
    df['Title'] = df['Title'].replace('Sir', 'Mr')
    df['Title'] = df['Title'].replace('the Countess', 'Mrs')

    # 創建家庭大小特徵，因為大家庭或單獨旅行可能影響生存機率
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 創建是否獨行的特徵
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0  # 如果家庭成員數大於1，設置為非獨行

    # 門票價格分段
    df['FareBand'] = pd.qcut(df['Fare'], 4)

    # 年齡分段，將乘客年齡分成5個區間
    df['AgeBand'] = pd.cut(df['Age'], 5)
```

## 5. 類別變數轉換:
```python
# 使用 get_dummies() 方法將類別變數轉換為虛擬變數，方便隨機森林模型進行訓練
train = pd.get_dummies(train, columns=['Sex', 'Embarked', 'Title', 'FareBand', 'AgeBand'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked', 'Title', 'FareBand', 'AgeBand'], drop_first=True)

# 保留 PassengerId 以便後續生成提交文件
passenger_ids = test['PassengerId']
train = train.drop(['Cabin', 'Name', 'Ticket'], axis=1)
test = test.drop(['Cabin', 'Name', 'Ticket'], axis=1)

# 確保 train 和 test 的特徵對齊，填補測試集中的缺失列
test = test.reindex(columns=train.columns.drop('Survived'), fill_value=0)
```

## 6. 數據分割:
```python
# 切分訓練集為特徵 (X) 和標籤 (y)
X = train.drop(['Survived'], axis=1)
y = train['Survived']

# 將數據分割為訓練集和測試集，以進行模型的評估
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 7. 構建隨機森林模型:
```python
# 初始化隨機森林分類器並訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測測試集的生存情況
y_pred = model.predict(X_test)

# 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
```

## 8. 使用 GridSearch 進行模型優化:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 使用 GridSearchCV 進行超參數調優
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# 使用最佳模型進行測試集預測
y_test_pred = best_model.predict(test)
```

## 9. 生成文件:
```python
# 最後，將預測結果寫入 CSV 文件，便於提交
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": y_test_pred
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
```
