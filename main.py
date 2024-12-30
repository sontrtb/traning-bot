from datetime import datetime, timedelta
from get_data import GetData
from model import FinancialPredictor

#Config
symbol="IOTAUSDT"
window_size=48
days_back=1
iterations=300
epochs=50

# Dự đoán 10 bước tiếp theo từ file
steps = 5

# Lấy dữ liệu về
def getDataTrain(days_back=days_back, iterations=iterations):
    today = datetime.today()
    for i in range(iterations):
        date = today - timedelta(days=days_back)
        dateFormat = date.strftime('%d/%m/%Y')
        print(dateFormat)
        fetcher = GetData(date=dateFormat, is_now=False, symbol=symbol)
        fetcher.get_data()
        days_back += 1

def getDataNow():
    fetcher = GetData(is_now=True, symbol=symbol)
    fetcher.get_data()


# Training
def traningModel():
    # Huấn luyện và lưu mô hình
    predictor = FinancialPredictor(window_size=window_size, epochs=epochs)
    predictor.load_multiple_files(f"data/{symbol}")
    predictor.preprocess_data()
    predictor.build_model()
    predictor.train()
    predictor.save_model(f"saved_model/{symbol}")

## Chay model
def runModelDataNow():
    predictor = FinancialPredictor.load_model(f"saved_model/{symbol}")
    result = predictor.predict_multiple_from_file(
        f"data_now/{symbol}/now.csv",
        steps=steps,
        include_history=True
    )
    print("\nGiá đóng cửa cuối cùng:", result['info']['last_close'])
    print("\nDự đoán cho các phiên tiếp theo:")
    for i, pred in enumerate(result['predictions']):
        print(f"Phiên {i+1}: Open={pred[0]:.4f}, Close={pred[3]:.4f}")
    predictor.plot_predictions(
        result['predictions'],
        result['history'],
        feature_names=['Price']
    )

# RUN
# getDataTrain()
# traningModel()

getDataNow()
runModelDataNow()
