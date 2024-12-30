# import schedule
# import time

# interval = 5

# def job():
#     print("Job đang chạy...")

# # Lên lịch chạy job mỗi 5 giây
# schedule.every(interval).seconds.do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

from forecast import Forecast

# Config
symbol="IOTAUSDT"
steps=5

forecast = Forecast(symbol=symbol, steps=steps)
forecastRes = forecast.runModelDataNow()

print("\n")
if(forecastRes["percent"] > 5):
    print("Vào lệnh: ", forecastRes["trend"])
else:
    print("Không vào lệnh")
