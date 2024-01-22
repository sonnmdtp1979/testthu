# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ file CSV
data = pd.read_csv('winequality-white.csv', sep=';')

# Chọn các cột bạn muốn sử dụng cho huấn luyện mô hình
selected_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']
X = data[selected_columns]
y = data['quality']

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá độ chính xác của mô hình trên tập kiểm thử
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Lưu mô hình vào file
joblib.dump(model, 'wine_quality_model.joblib')
