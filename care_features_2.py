def care_features(ct):
    if ct=='New':

        care_cont_features=['AXC_CARE','ENTERING_CONTACT_US_EXPERIENCE','TOTAL_AGENT_INTERACTION_SECONDS']

        care_bool_features=['AXC_CARE','ENTERING_CONTACT_US_EXPERIENCE']
        care_catag_features=['CARE_REFERRER',	'FIRST_HS_INTENT',	'FIRST_SEARCH_MANUAL_LOCATION_DETAIL']

        care_cont_score_features = ['AXC_CARE','ENTERING_CONTACT_US_EXPERIENCE','TOTAL_AGENT_INTERACTION_SECONDS']

        care_bool_score_features = ['AXC_CARE', 'ENTERING_CONTACT_US_EXPERIENCE']
        care_catag_score_features = ['CARE_REFERRER',	'FIRST_HS_INTENT',	'FIRST_SEARCH_MANUAL_LOCATION_DETAIL']


    return care_cont_features, care_bool_features, care_catag_features, care_cont_score_features, care_bool_score_features, care_catag_score_features
