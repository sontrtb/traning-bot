import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import keras
import matplotlib.pyplot as plt
import glob
import os
import joblib
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import sklearn.exceptions

class FinancialPredictor:
    def __init__(self, window_size=96, epochs=150):
        self.window_size = window_size
        self.epochs = epochs
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.indicator_scaler = RobustScaler()
        self.scaler = RobustScaler()
        self.model = None
        self.training_history = []
        
    
    def update_model(self, new_data_path, epochs=None, save_dir=None):
        """Cập nhật mô hình với dữ liệu mới"""
        print("Bắt đầu quá trình cập nhật mô hình...")
        
        # Đọc dữ liệu mới
        if isinstance(new_data_path, str):
            # Nếu là đường dẫn đến một file
            new_df = pd.read_csv(new_data_path)
            new_df = new_df.drop(["Time"], axis='columns')
        elif isinstance(new_data_path, list):
            # Nếu là danh sách các file
            new_df = self.load_selected_files(new_data_path)
        else:
            raise ValueError("new_data_path phải là đường dẫn file hoặc danh sách các file")

        # Chuẩn bị dữ liệu mới
        X_new, y_new = self.create_sequences(new_df)
        X_new_scaled = self.scaler.transform(X_new.reshape(-1, X_new.shape[-1])).reshape(X_new.shape)
        y_new_scaled = self.scaler.transform(y_new)

        # Cập nhật mô hình với dữ liệu mới
        update_epochs = epochs if epochs is not None else self.epochs
        print(f"Cập nhật mô hình với {len(X_new)} mẫu dữ liệu mới...")
        
        history = self.model.fit(
            X_new_scaled,
            y_new_scaled,
            epochs=update_epochs,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        # Lưu thông tin về lần cập nhật này
        update_info = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': new_data_path if isinstance(new_data_path, str) else 'multiple_files',
            'num_samples': len(X_new),
            'epochs': update_epochs,
            'final_loss': history.history['loss'][-1],
            'final_mae': history.history['mae'][-1]
        }
        self.training_history.append(update_info)

        # Tự động lưu mô hình nếu được chỉ định
        if save_dir:
            self.save_model(save_dir)

        print("\nKết quả cập nhật:")
        print(f"- Số lượng mẫu: {len(X_new)}")
        print(f"- Loss cuối cùng: {update_info['final_loss']:.4f}")
        print(f"- MAE cuối cùng: {update_info['final_mae']:.4f}")
        
        return history
    
    def load_multiple_files(self, directory_path, pattern="*.csv"):
        """Đọc dữ liệu từ nhiều file CSV trong một thư mục"""
        all_data = []
        file_paths = glob.glob(os.path.join(directory_path, pattern))
        
        if not file_paths:
            raise ValueError(f"Không tìm thấy file nào khớp với pattern {pattern} trong thư mục {directory_path}")
        
        print(f"Đang đọc {len(file_paths)} files:")

        files_with_time = [(file, os.path.getctime(file)) for file in file_paths]

        # Sắp xếp các tệp theo ngày tạo (ctime)
        files_sorted = sorted(files_with_time, key=lambda x: x[1], reverse=True)
        for file_path, _ in files_sorted:
            print(f"- Đang đọc file: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            # Loại bỏ cột thời gian
            df = df.drop(["Time"], axis='columns')
            all_data.append(df)
            
        # Gộp tất cả dataframes
        self.data = pd.concat(all_data, axis=0, ignore_index=True)
        print(f"Tổng số dòng dữ liệu: {len(self.data)}")
        return self.data
    
    def load_selected_files(self, file_paths):
        """Đọc dữ liệu từ danh sách các file được chọn"""
        all_data = []
        
        print(f"Đang đọc {len(file_paths)} files:")
        for file_path in file_paths:
            print(f"- Đang đọc file: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            # Loại bỏ cột thời gian
            df = df.drop(["Time"], axis='columns')
            all_data.append(df)
            
        # Gộp tất cả dataframes
        self.data = pd.concat(all_data, axis=0, ignore_index=True)
        print(f"Tổng số dòng dữ liệu: {len(self.data)}")
        return self.data
    
    def create_sequences(self, data):
        """Tạo chuỗi dữ liệu cho LSTM"""
        values = data.values
        X = []
        y = []
        for i in range(len(values) - self.window_size):
            X.append(values[i:i+self.window_size])
            y.append(values[i+self.window_size])
        return np.array(X), np.array(y)
    
    def preprocess_data(self, add_technical_indicators=True):
        """Tiền xử lý dữ liệu với các chỉ báo kỹ thuật bổ sung"""
        if add_technical_indicators:
            self.add_technical_indicators()
            
        X, y = self.create_sequences(self.data)
        
        # Tách dữ liệu thành các thành phần để xử lý riêng
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_cols = ['Volume']
        indicator_cols = [col for col in self.data.columns if col not in price_cols + volume_cols]
        
        # Reshape và scale dữ liệu
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Scale từng nhóm features riêng biệt
        X_price = self.price_scaler.fit_transform(X_reshaped[:, :4])
        X_volume = self.volume_scaler.fit_transform(X_reshaped[:, 4:5])
        if len(indicator_cols) > 0:
            X_indicators = self.indicator_scaler.fit_transform(X_reshaped[:, 5:])
            # Kết hợp lại
            X_scaled = np.concatenate([X_price, X_volume, X_indicators], axis=1)
        else:
            X_scaled = np.concatenate([X_price, X_volume], axis=1)
        
        X_scaled = X_scaled.reshape(X.shape)
        
        # Scale chỉ giá OHLC cho target
        y_scaled = self.price_scaler.transform(y[:, :4])
        
        self.X_scaled = X_scaled
        self.y_scaled = y_scaled
        self.n_features = X_scaled.shape[2]
        return self.X_scaled, self.y_scaled

    def add_technical_indicators(self):
        """Thêm các chỉ báo kỹ thuật phức tạp"""
        df = self.data.copy()
        
        # Exponential Moving Averages
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # Bollinger Bands
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        self.data = df
        return df

    def build_model(self):
        """Xây dựng mô hình LSTM với kiến trúc nâng cao"""
        input_shape = (self.window_size, self.n_features)
        
        self.model = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(192, activation='relu', 
                             input_shape=input_shape,
                             return_sequences=True)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Bidirectional(keras.layers.LSTM(96, activation='relu', return_sequences=True)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Bidirectional(keras.layers.LSTM(48, activation='relu')),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(48, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(24, activation='relu'),
            keras.layers.BatchNormalization(),
            
            # Đầu ra chỉ có 4 giá trị cho OHLC
            keras.layers.Dense(4)
        ])

        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.5
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mape']
        )
        return self.model

    def train(self):
        """Huấn luyện mô hình với các callbacks nâng cao"""
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            mode='min'
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
        
        best_val_loss = float('inf')
        best_model = None
        
        # Training với Time Series Cross Validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_scaled)):
            print(f'\nFold {fold + 1}')
            X_train, X_val = self.X_scaled[train_idx], self.X_scaled[val_idx]
            y_train, y_val = self.y_scaled[train_idx], self.y_scaled[val_idx]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=64,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = keras.models.clone_model(self.model)
                best_model.set_weights(self.model.get_weights())
        
        self.model = best_model
        return history
    
    def predict(self, window_data=None):
        """Dự đoán giá trị tiếp theo"""
        if window_data is None:
            window_data = self.X_scaled[-1]
        
        prediction = self.model.predict(window_data.reshape(1, self.window_size, -1))
        # Chuyển đổi về giá trị gốc
        prediction_original = self.scaler.inverse_transform(prediction)
        return prediction_original

    def predict_from_file(self, file_path):
        """Dự đoán từ dữ liệu trong một file mới"""
        # Đọc và xử lý file mới
        df = pd.read_csv(file_path)
        df = df.drop(["Time"], axis='columns')
        
        # Lấy cửa sổ dữ liệu cuối cùng
        values = df.values
        if len(values) < self.window_size:
            raise ValueError(f"File cần ít nhất {self.window_size} dòng dữ liệu")
            
        window_data = values[-self.window_size:]
        
        # Chuẩn hóa dữ liệu
        window_scaled = self.scaler.transform(window_data)
        
        # Dự đoán
        prediction = self.predict(window_scaled)
        return prediction
    
    def save_model(self, directory_path):
        """Lưu mô hình, scaler và các thông số"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        # Lưu mô hình Keras
        model_path = os.path.join(directory_path, 'model.h5')
        self.model.save(model_path)
        
        # Lưu scaler
        scaler_path = os.path.join(directory_path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Lưu các thông số và lịch sử training
        params = {
            'window_size': self.window_size,
            'epochs': self.epochs,
            'training_history': self.training_history
        }
        params_path = os.path.join(directory_path, 'params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        print(f"Đã lưu mô hình tại: {directory_path}")
        print(f"- Model: {model_path}")
        print(f"- Scaler: {scaler_path}")
        print(f"- Parameters và lịch sử: {params_path}")

    @classmethod
    def load_model(cls, directory_path):
        """Tải mô hình, scaler và các thông số"""
        if not os.path.exists(directory_path):
            raise ValueError(f"Không tìm thấy thư mục: {directory_path}")
            
        # Tải các thông số
        params_path = os.path.join(directory_path, 'params.json')
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        # Tạo instance mới
        instance = cls(
            window_size=params['window_size'],
            epochs=params['epochs']
        )
        
        # Tải mô hình và scaler
        model_path = os.path.join(directory_path, 'model.h5')
        instance.model = keras.models.load_model(model_path)
        
        scaler_path = os.path.join(directory_path, 'scaler.pkl')
        instance.scaler = joblib.load(scaler_path)
        
        # Tải lịch sử training
        if 'training_history' in params:
            instance.training_history = params['training_history']
        
        print(f"\nĐã tải mô hình từ: {directory_path}")
            
        return instance
    
    def predict_multiple(self, steps, window_data=None, include_history=False, confidence_interval=True):
        if window_data is None:
            window_data = self.X_scaled[-1]
        
        window_data = window_data.reshape(1, self.window_size, self.n_features)
        predictions = []
        confidence_intervals = []
        current_window = window_data.copy()
        n_simulations = 100
        
        for step in range(steps):
            step_predictions = []
            
            for _ in range(n_simulations):
                input_data = current_window.reshape(1, self.window_size, self.n_features)
                next_pred = self.model.predict(input_data, verbose=0)
                step_predictions.append(next_pred[0])
            
            step_predictions = np.array(step_predictions)
            mean_pred = np.mean(step_predictions, axis=0)
            std_pred = np.std(step_predictions, axis=0)
            
            confidence_interval = {
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            }
            
            predictions.append(mean_pred)
            confidence_intervals.append(confidence_interval)
            
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1, :] = mean_pred
        
        predictions = np.array(predictions)
        # Chuyển predictions về giá trị gốc
        predictions_original = self.price_scaler.inverse_transform(predictions[:, :4])
        
        if confidence_interval:
            ci_lower = np.array([ci['lower'] for ci in confidence_intervals])
            ci_upper = np.array([ci['upper'] for ci in confidence_intervals])
            ci_lower_original = self.price_scaler.inverse_transform(ci_lower[:, :4])
            ci_upper_original = self.price_scaler.inverse_transform(ci_upper[:, :4])
            
            if include_history:
                history_original = self.price_scaler.inverse_transform(window_data[0, :, :4])
                return predictions_original, ci_lower_original, ci_upper_original, history_original
            
            return predictions_original, ci_lower_original, ci_upper_original
        
        if include_history:
            history_original = self.price_scaler.inverse_transform(window_data[0, :, :4])
            return predictions_original, history_original
        
        return predictions_original
   
    def predict_multiple_from_file(self, file_path, steps, include_history=False):
        # Đọc và xử lý file
        df = pd.read_csv(file_path)
        df = df.drop(["Time"], axis='columns')
        
        if len(df) < self.window_size:
            raise ValueError(f"File cần ít nhất {self.window_size} dòng dữ liệu")
            
        # Lấy cửa sổ dữ liệu cuối cùng
        window_data = df.values[-self.window_size:]
        last_close = window_data[-1][3]
        
        # Xử lý scaler chưa được fit
        try:
            window_scaled = self.scaler.transform(window_data)
        except sklearn.exceptions.NotFittedError:
            # Fit scaler với dữ liệu hiện tại nếu chưa được fit
            self.scaler.fit(window_data)
            window_scaled = self.scaler.transform(window_data)
        
        if include_history:
            predictions, history = self.predict_multiple(steps, window_scaled, include_history=True)
            predictions[0][0] = last_close
            
            return {
                'predictions': predictions,
                'history': history,
                'info': {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'steps': steps,
                    'source_file': os.path.basename(file_path),
                    'window_size': self.window_size,
                    'last_close': last_close
                }
            }
        
        predictions = self.predict_multiple(steps, window_scaled)
        predictions[0][0] = last_close
        
        return {
            'predictions': predictions,
            'info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'steps': steps,
                'source_file': os.path.basename(file_path),
                'window_size': self.window_size,
                'last_close': last_close
            }
        }

    def plot_training_metrics(self):
        """Vẽ đồ thị các metrics qua các lần cập nhật"""
        if not self.training_history:
            print("Chưa có lịch sử training")
            return
            
        losses = [update['final_loss'] for update in self.training_history]
        maes = [update['final_mae'] for update in self.training_history]
        updates = range(1, len(self.training_history) + 1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(updates, losses, 'b-o')
        plt.title('Loss qua các lần cập nhật')
        plt.xlabel('Lần cập nhật')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(updates, maes, 'r-o')
        plt.title('MAE qua các lần cập nhật')
        plt.xlabel('Lần cập nhật')
        plt.ylabel('MAE')
        
        plt.tight_layout()
        plt.show()

    
# Huấn luyện và lưu mô hình
# predictor = FinancialPredictor(window_size=100, epochs=100)
# data = predictor.load_multiple_files("data")
# X_scaled, y_scaled = predictor.preprocess_data()
# model = predictor.build_model()
# history = predictor.train()
# predictor.save_model("saved_model")

# # Sau này, tải và sử dụng mô hình đã lưu
# loaded_predictor = FinancialPredictor.load_model("saved_model")
# prediction = loaded_predictor.predict_from_file("data/data_test.csv")

# Ví dụ sử dụng Update Model:

# # 1. Tải mô hình đã huấn luyện
# predictor = FinancialPredictor.load_model("saved_model")

# # 2. Cập nhật với một file dữ liệu mới
# history = predictor.update_model(
#     new_data_path="new_data.csv",
#     epochs=50,
#     save_dir="updated_model"
# )

# 3. Cập nhật với nhiều file
# new_files = ["data1.csv", "data2.csv", "data3.csv"]
# history = predictor.update_model(
#     new_data_path=new_files,
#     epochs=50,
#     save_dir="updated_model"
# )

# 4. Xem lịch sử các lần cập nhật
# predictor.plot_training_metrics()


# Tải mô hình
# predictor = FinancialPredictor.load_model("saved_model")

# # Dự đoán 10 bước tiếp theo từ file
# steps = 10
# result = predictor.predict_multiple_from_file(
#     "data/data_test.csv",
#     steps=10,
#     include_history=True
# )

# print("\nKêt quả {steps} bước tiếp theo:\n")
# print(result['predictions'])

# # Vẽ đồ thị kết quả
# predictor.plot_predictions(
#     result['predictions'],
#     result['history'],
#     feature_names=['Price', 'Volume', 'RSI']
# )