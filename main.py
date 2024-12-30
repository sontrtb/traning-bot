import schedule
import time
from forecast import Forecast
from traning import Traning
from colorama import Fore, Style

# Config
symbol="IOTAUSDT"
steps=5

interval = 900

traning = Traning(symbol)

def getDataTrain():
    traning.getDataTrain()

def traningModel():
    traning.traningModel()

def run():
    forecast = Forecast(symbol=symbol, steps=steps)
    forecastRes = forecast.runModelDataNow()

    print("\n")
    if(forecastRes["percent"] > 5):
        print(Style.BRIGHT + Fore.BLUE + "Vào lệnh: ", forecastRes["trend"])
    else:
        print(Style.BRIGHT + Fore.RED + "Không vào lệnh")
    
    print("\n")


# getDataTrain()
traningModel()

# run()

# Lên lịch chạy job mỗi {interval} giây
schedule.every(interval).seconds.do(run)

while True:
    schedule.run_pending()
    time.sleep(1)