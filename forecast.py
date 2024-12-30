from datetime import datetime, timedelta
from get_data import GetData
from model import FinancialPredictor

class Forecast:
    def __init__(self, symbol="IOTAUSDT", steps=5):
        self.symbol = symbol
        self.steps = steps

    def __getDataNow(self):
        fetcher = GetData(is_now=True, symbol=self.symbol)
        fetcher.get_data()

    def runModelDataNow(self):
        self.__getDataNow()
        
        predictor = FinancialPredictor.load_model(f"saved_model/{self.symbol}")
        result = predictor.predict_multiple_from_file(
            f"data_now/{self.symbol}/now.csv",
            steps=self.steps,
            include_history=True
        )
        print("\nGiá đóng cửa cuối cùng:", result['info']['last_close'], ". Vào lúc", datetime.now().strftime("%H:%M:%S %d-%m-%y"))
        print("\nDự đoán cho các phiên tiếp theo:")
        for i, pred in enumerate(result['predictions']):
            print(f"Phiên {i+1}: Open={pred[0]:.4f}, Close={pred[3]:.4f}")

        # predictor.plot_predictions(
        #     result['predictions'],
        #     result['history'],
        #     feature_names=['Price']
        # )
        
        open = result['predictions'][0][0]
        close = result['predictions'][self.steps - 1][3]
        percent = float(abs(((close - open) / open) * 100))

        print("\nDự đoán giá cuối : ", round(close, 4))
        print("Tăng" if open < close else "Giảm", ": ", round(percent, 2), "%")

        return {
            'trend': "UP" if open < close else "DOWN",
            'percent': percent,
        }
