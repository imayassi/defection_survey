for x in range(500001,5000000, 500000):
    i=x
    j=i+500000-1
    query="SELECT * FROM (SELECT *, ROW_NUMBER() OVER() AS RANK FROM  CTG_ANALYTICS_WS.SM_TXML_TY13_TY14_S_TY15 where  )A WHERE RANK BETWEEN %d AND %d  limit 1" % (i,j)
    # print  'new_customer_defection_prediction_%d.csv'%(i)
    print query
