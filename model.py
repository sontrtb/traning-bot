import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
import matplotlib.pyplot as plt
import glob
import os
import joblib
import json
from datetime import datetime

class FinancialPredictor:
    def __init__(self, window_size=100, epochs=100):
        self.window_size = window_size
        self.epochs = epochs
        self.scaler = MinMaxScaler()
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
            batch_size=32,
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
    
    def preprocess_data(self):
        """Tiền xử lý dữ liệu"""
        X, y = self.create_sequences(self.data)
        # Chuẩn hóa dữ liệu
        self.X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.y_scaled = self.scaler.transform(y)
        return self.X_scaled, self.y_scaled
    
    def build_model(self):
        """Xây dựng mô hình LSTM"""
        self.model = keras.Sequential([
            keras.layers.LSTM(128, activation='relu', 
                            input_shape=(self.X_scaled.shape[1], self.X_scaled.shape[2]), 
                            return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.X_scaled.shape[2])
        ])

        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
        )
        
        self.model.compile(optimizer=optimizer, 
                          loss='mean_squared_error', 
                          metrics=['mae'])
        return self.model
    
    def train(self):
        """Huấn luyện mô hình"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            self.X_scaled, 
            self.y_scaled,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
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
    
    def predict_multiple(self, steps, window_data=None, include_history=False):
        """Dự đoán nhiều giá trị tiếp theo với đảm bảo tính liên tục giữa các phiên

        Parameters:
        -----------
        steps : int
            Số lượng bước dự đoán phía trước
        window_data : numpy.ndarray, optional
            Dữ liệu cửa sổ đầu vào. Nếu không cung cấp, sẽ sử dụng cửa sổ cuối cùng
        include_history : bool, optional
            Nếu True, trả về cả dữ liệu lịch sử của cửa sổ đầu vào

        Returns:
        --------
        numpy.ndarray
            Mảng các giá trị dự đoán
        numpy.ndarray, optional
            Dữ liệu lịch sử (nếu include_history=True)
        """
        if window_data is None:
            window_data = self.X_scaled[-1]
        
        # Chuẩn bị mảng để lưu kết quả
        predictions = []
        current_window = window_data.copy()
        
        # Thực hiện dự đoán tuần tự
        for step in range(steps):
            # Dự đoán giá trị tiếp theo
            next_pred = self.model.predict(current_window.reshape(1, self.window_size, -1), verbose=0)
            
            # Đảm bảo tính liên tục giữa các phiên
            if step > 0:
                # Giá mở cửa của phiên hiện tại = giá đóng cửa của phiên trước
                next_pred[0][0] = predictions[-1][3]  # Assuming index 3 is Close price
            
            predictions.append(next_pred[0])
            
            # Cập nhật cửa sổ cho lần dự đoán tiếp theo
            current_window = np.roll(current_window, -1, axis=0)
            current_window[-1] = next_pred[0]

        # Chuyển đổi kết quả về giá trị gốc
        predictions = np.array(predictions)
        predictions_original = self.scaler.inverse_transform(predictions)
        
        if include_history:
            history_original = self.scaler.inverse_transform(window_data)
            return predictions_original, history_original
        
        return predictions_original

    def predict_multiple_from_file(self, file_path, steps, include_history=False):
        """Dự đoán nhiều giá trị tiếp theo từ dữ liệu trong file với đảm bảo tính liên tục

        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file dữ liệu
        steps : int
            Số lượng bước dự đoán phía trước
        include_history : bool, optional
            Nếu True, trả về cả dữ liệu lịch sử của cửa sổ đầu vào

        Returns:
        --------
        dict
            Kết quả dự đoán và thông tin liên quan
        """
        # Đọc và xử lý file
        df = pd.read_csv(file_path)
        df = df.drop(["Time"], axis='columns')
        
        # Kiểm tra dữ liệu đầu vào
        if len(df) < self.window_size:
            raise ValueError(f"File cần ít nhất {self.window_size} dòng dữ liệu")
            
        # Lấy cửa sổ dữ liệu cuối cùng
        window_data = df.values[-self.window_size:]
        
        # Lấy giá đóng cửa cuối cùng để đảm bảo tính liên tục
        last_close = window_data[-1][3]  # Assuming index 3 is Close price
        
        # Chuẩn hóa dữ liệu
        window_scaled = self.scaler.transform(window_data)
        
        # Thực hiện dự đoán
        if include_history:
            predictions, history = self.predict_multiple(steps, window_scaled, include_history=True)
            
            # Đảm bảo giá mở cửa đầu tiên = giá đóng cửa cuối cùng của lịch sử
            predictions[0][0] = last_close
            
            result = {
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
            return result
        else:
            predictions = self.predict_multiple(steps, window_scaled)
            
            # Đảm bảo giá mở cửa đầu tiên = giá đóng cửa cuối cùng
            predictions[0][0] = last_close
            
            result = {
                'predictions': predictions,
                'info': {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'steps': steps,
                    'source_file': os.path.basename(file_path),
                    'window_size': self.window_size,
                    'last_close': last_close
                }
            }
            return result

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

# 3. Cập nhật với nhiều file
# new_files = ["data1.csv", "data2.csv", "data3.csv"]
# history = predictor.update_model(
#     new_data_path=new_files,
#     epochs=50,
#     save_dir="updated_model"
# )

# 4. Xem lịch sử các lần cập nhật
# predictor.plot_training_metrics()
