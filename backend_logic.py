import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def load_data(filepath):
    """Tải dữ liệu từ file CSV."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def preprocess_data(df, handle_outliers=True, scaler_choice="RobustScaler"):
    """Hàm tiền xử lý dữ liệu tổng hợp với thứ tự đúng."""
    # 1. Loại bỏ các cột không cần thiết
    drop_list = ['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood']
    df = df.drop(drop_list, axis=1)

    # 2. Xử lý giá trị thiếu bằng KNNImputer
    if df['reviews_per_month'].isnull().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df['reviews_per_month'] = imputer.fit_transform(df[['reviews_per_month']])

    # Xác định các cột số để xử lý
    # Quan trọng: Bao gồm cả 'price' để xử lý ngoại lai cho nó trước khi tạo đặc trưng
    num_cols_for_outliers = df.select_dtypes(include=np.number).columns.tolist()

    # 3. Xử lý giá trị ngoại lai (tùy chọn) - ĐƯA LÊN TRƯỚC FEATURE ENGINEERING
    if handle_outliers:
        for col in num_cols_for_outliers:
            # Không xử lý ngoại lai cho các cột đã one-hot nếu có
            if df[col].nunique() > 2: # Bỏ qua các cột nhị phân
                 replace_with_thresholds(df, col)
    
    # 4. Trích xuất đặc trưng mới (bây giờ sẽ dùng dữ liệu đã được xử lý ngoại lai)
    df['NEW_total_cost'] = df['price'] * df['minimum_nights']
    df['NEW_availability_ratio'] = df['availability_365'] / 365
    df['NEW_annual_income'] = df['price'] * df['availability_365']
    
    # 5. Mã hóa biến phân loại bằng One-Hot Encoding
    cat_cols = ['neighbourhood_group', 'room_type']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 6. Tách X và y
    y = df['price']
    X = df.drop('price', axis=1)

    # Xác định lại các cột số trong X để chuẩn hóa
    num_cols_for_scaling = X.select_dtypes(include=np.number).columns.tolist()
    
    # 7. Chuẩn hóa dữ liệu (tùy chọn)
    if scaler_choice == "RobustScaler":
        scaler = RobustScaler()
    elif scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    else: # Mặc định hoặc None
        scaler = None

    if scaler and num_cols_for_scaling:
        X[num_cols_for_scaling] = scaler.fit_transform(X[num_cols_for_scaling])
        
    # Xử lý các tên cột không hợp lệ cho một số mô hình
    X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

    return X, y

def get_model(model_name, params):
    """Khởi tạo mô hình với các siêu tham số."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=params.get('alpha', 1.0)),
        'Lasso': Lasso(alpha=params.get('alpha', 1.0)),
        'ElasticNet': ElasticNet(alpha=params.get('alpha', 1.0), l1_ratio=params.get('l1_ratio', 0.5)),
        'KNN': KNeighborsRegressor(n_neighbors=params.get('n_neighbors', 5)),
        'Decision Tree': DecisionTreeRegressor(max_depth=params.get('max_depth', None), min_samples_leaf=params.get('min_samples_leaf', 1)),
        'Random Forest': RandomForestRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None)),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 0.1)),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 0.1)),
        'LightGBM': LGBMRegressor(n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 0.1)),
        'CatBoost': CatBoostRegressor(iterations=params.get('iterations', 100), learning_rate=params.get('learning_rate', 0.1), verbose=False)
    }
    return models.get(model_name)

def train_and_evaluate(X, y, model_name, params):
    """Huấn luyện và đánh giá mô hình."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = get_model(model_name, params)
    
    if model is None:
        raise ValueError("Model không hợp lệ!")
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Lọc giá trị 0 để tính MAPE
    mask = y_test != 0
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]
    
    # Tính toán metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R^2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test_filtered, y_pred_filtered) * 100 if len(y_test_filtered) > 0 else np.nan
    }
    
    return metrics