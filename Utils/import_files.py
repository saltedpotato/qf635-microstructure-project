import requests
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np

logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', level=logging.INFO)
BASE_URL = "https://api.binance.com/api/v3"
BASE_URL_ORDER_BOOK = "https://fapi.binance.com/fapi/v1"

