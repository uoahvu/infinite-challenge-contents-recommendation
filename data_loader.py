import pandas as pd
import numpy as np
from datetime import datetime


def timestr2min(x):
    splited_time = x.split(":")
    return (
        float(splited_time[2]) / 60.0
        + float(splited_time[1])
        + float(splited_time[0]) * 60.0
    )


def date2season(x):
    periods = [
        ("2006-05-06", "2008-03-08", 1),  # 2006.05.06 ~ 2008.03.08
        ("2008-03-15", "2009-05-30", 2),  # 2008.03.15 ~ 2009.05.30 +전진 -하하
        ("2009-06-06", "2010-03-20", 3),  # 2009.06.06 ~ 2010.03.20 +길 -전진
        ("2010-03-27", "2012-01-31", 4),  # 2010.03.27 ~ 2012.01 +하하 (7인)
        # ("2012-02-01", "2012-07-14", 5),  # 2012.02 ~ 2012.07.14 파업
        ("2012-07-14", "2014-05-03", 5),  # 2012.07.14 ~ 2014.05.03 (7인)
        ("2014-05-03", "2014-11-15", 6),  # 2014.05.03 ~ 2014.11.08 -길
        ("2014-11-17", "2015-05-09", 7),  # 2014.11.17 ~ 2015.05.09 -홍철
        ("2015-05-09", "2015-11-12", 8),  # 2015.05.09 ~ 2015.11.12 +광희
        ("2015-11-12", "2016-04-09", 9),  # 2015.11.12 ~ 2016.04.09 -형돈
        ("2016-04-09", "2017-03-25", 10),  # 2016.04.09 ~ 2017.03.25 +세형
        ("2017-03-25", "2017-09-08", 11),  # 2017.03.25 ~ 2017.09.08 -광희
        ("2017-09-08", "2017-11-18", 12),  # 2017.09.08 ~ 2017.11.18 파업
        ("2017-11-25", "2018-03-31", 13),  # 2017.11.25 ~ 2018.03.31 +세호
    ]
    for start_date, end_date, season in periods:
        if x <= end_date and x >= start_date:
            return season
    return 0


def data_preprocess():
    df = pd.read_csv("./datasets/무한도전_전회차_크롤링_데이터셋.csv")

    # Description
    df["description"] = df["description"].str.replace(
        r"[^a-zA-Z가-힣0-9 ]", "", regex=True
    )

    # RunTime
    df["time2min"] = df["time"].apply(lambda x: timestr2min(x))

    # Date
    df["season"] = df["date"].apply(lambda x: date2season(x))

    # Title
    df["title_"] = df["title"].str.replace(r"[무한도전]", "")

    # 특집회 여부
    df["special"] = df["vod_num"].apply(lambda x: 1 if x == "특집회" else 0)
    return df
