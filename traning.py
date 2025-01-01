from datetime import datetime, timedelta
from get_data import GetData
from model import FinancialPredictor

class Traning:
    def __init__(self, symbol="IOTAUSDT"):
        self.symbol = symbol
        self.window_size = 96
        self.days_back = 1
        self.iterations = 300
        self.epochs = 50
    
    # Lấy dữ liệu về
    def getDataTrain(self):
        today = datetime.today()
        for i in range(self.iterations):
            date = today - timedelta(days=self.days_back)
            dateFormat = date.strftime('%d/%m/%Y')
            print(dateFormat)
            fetcher = GetData(date=dateFormat, is_now=False, symbol=self.symbol)
            fetcher.get_data()
            self.days_back += 1

    # Training
    def traningModel(self):
        # Huấn luyện và lưu mô hình
        predictor = FinancialPredictor(window_size=self.window_size, epochs=self.epochs)
        predictor.load_multiple_files(f"data/{self.symbol}")
        predictor.preprocess_data()
        predictor.build_model()
        predictor.train()
        predictor.save_model(f"saved_model/{self.symbol}")