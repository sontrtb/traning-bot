import csv
from datetime import datetime, timedelta
import os
import ccxt
import pytz
import logging

class GetData:
    def __init__(self, date=None, is_now=True, symbol="ADA/USDT"):
        self.date = date or datetime.now().strftime('%d/%m/%Y')
        self.is_now = is_now
        self.timezone = pytz.timezone("Asia/Bangkok")
        self.exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})
        self.symbol = symbol

    def _convert_timestamp(self, timestamp):
        """Chuyển đổi timestamp sang định dạng ngày giờ."""
        utc_time = datetime.utcfromtimestamp(timestamp / 1000)
        local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(self.timezone)
        return local_time.strftime('%H:%M:%S %d/%m/%Y')

    def get_data(self):
        symbol = self.symbol
        timeframe = '15m'
        limit = 96

        try:
            # Lấy dữ liệu từ API Binance
            since = None if self.is_now else int(datetime.strptime(self.date, '%d/%m/%Y').timestamp() * 1000)
            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
        except Exception as e:
            logging.error(f"Lỗi khi lấy dữ liệu từ Binance: {e}")
            return

        # Chuyển đổi timestamp
        for row in data:
            row[0] = self._convert_timestamp(row[0])

        # Tạo thư mục lưu trữ nếu chưa có
        directory = f"data_now/{symbol}" if self.is_now else f"data/{symbol}"
        os.makedirs(directory, exist_ok=True)
        filename = "now.csv" if self.is_now else f"{self.date.replace('/', '-')}.csv"
        file_path = os.path.join(directory, filename)

        # Ghi dữ liệu vào file CSV
        header = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(data)
            logging.info(f"File CSV đã được tạo: {file_path}")
        except Exception as e:
            logging.error(f"Lỗi khi ghi dữ liệu vào file CSV: {e}")

# # Sử dụng lớp GetData
# fetcher = GetData(date="18/12/2024", is_now=False)  # Định dạng dd/mm/yyyy
# # fetcher = GetData()
# fetcher.get_data()
