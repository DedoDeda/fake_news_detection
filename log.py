import datetime as dt


LOG_FORMAT = '%H:%M:%S'


def log(message):
    print(f'[{dt.datetime.now().strftime(LOG_FORMAT)}] {message}')
