import pyodbc
import pandas as pd
conn = pyodbc.connect(dsn='VerticaProd')

df2=pd.DataFrame()
# print df2
for x in range(150,12,-1):
    y=x
    print x
    # data="SELECT %d AS TAX_DAY, TAX_YEAR,PRODUCT_ROLLUP,CUSTOMER_TYPE, COUNT(DISTINCT RETAINED) AS RETAINED from  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_TEST_FULL WHERE CORE_FLAG=1 AND TTO_FLAG=1 AND TAX_DAY_CY=%d GROUP BY 1,2,3,4" % (x,y)
    # df = pd.read_sql(data, conn, coerce_float=False)
    # df2=pd.concat([df, df2], ignore_index=True)


    data="select COUNT (customer_key) as retained from CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_TEST_FULL where tax_year=2015 and tto_flag=1 and core_flag=1 and tax_day_cy<=%d and retained is not null and risk_flag=1" % x
    df = pd.read_sql(data, conn, coerce_float=False)
    df2=pd.concat([df, df2], ignore_index=True)


df2.to_csv(path_or_buf='at_risk_by_day.txt', index=True)