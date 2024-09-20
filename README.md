# Kaggle - Titanic 

### 1.載入所需套件與數據:
	
### 程式碼:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mp

#載入train test
train = pd.read_csv(r"train.csv")
test = pd.read_csv(r"test.csv")
```

### 確認缺失值
訓練集有 891 筆資料、12 個欄位
Age Cabin Embarked 有缺失值	

測試集有 418 筆資料、11 個欄位
Age Fare Cabin 有缺失值

### 程式碼:
```python
train.info()
test.info()
```
### 執行結果:
![image](--------------------------------------------------------0-1)
![image](--------------------------------------------------------0-2)

### 程式碼:
```python
D1 = train.describe()
D2 = test.describe()
```
### 執行結果:
![image](--------------------------------------------------------0-1)
![image](--------------------------------------------------------0-2)

#合併資料
data = pd.concat([train, test], ignore_index=True)
```
![image](---------------------------------------------------------)
