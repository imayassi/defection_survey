def care_features(ct):
    if ct=='New':
        care_cont_features=['AXC_CARE','CALL_SMARTLOOK_METRIC','WRAPUP_CODE_CNT','HANDLED_CNT', 'IVR_FAILURE_CNT','ENTERING_CONTACT_US_EXPERIENCE',	'WEB_HELP',	'VISIT_NUM','TOTAL_AGENT_INTERACTION_SECONDS','VIDEO_FLAG']

        care_bool_features=['AXC_CARE','CALL_SMARTLOOK_METRIC',	'ENTERING_CONTACT_US_EXPERIENCE',	'WEB_HELP',	'VIDEO_FLAG']
        care_catag_features=['CARE_REFERRER',	'FIRST_HS_INTENT',	'FIRST_CONTACT_CHANNEL',	'ANS_CR_VAL',	'ANS_EXP_VAL',	'ANS_NP_VAL',	'ANS_TT_CARES_VAL',	'ANS_CONF_CORR_VAL',	'ANS_EASY_RESOLVE_VAL',	'COBROWSE_FLAG',	'FIRST_SEARCH_MANUAL_LOCATION_DETAIL',	'SKILL_ID',	'CHANNEL_CARE',	'SRS_CATEGORY',	'WORKGROUP',	'WAIT_TIME']
        care_cont_score_features = ['AXC_CARE', 'CALL_SMARTLOOK_METRIC', 'WRAPUP_CODE_CNT', 'HANDLED_CNT', 'IVR_FAILURE_CNT',
                              'ENTERING_CONTACT_US_EXPERIENCE', 'WEB_HELP', 'VISIT_NUM',
                              'TOTAL_AGENT_INTERACTION_SECONDS', 'VIDEO_FLAG']

        care_bool_score_features = ['AXC_CARE', 'CALL_SMARTLOOK_METRIC', 'ENTERING_CONTACT_US_EXPERIENCE', 'WEB_HELP',
                              'VIDEO_FLAG']
        care_catag_score_features = ['CARE_REFERRER', 'FIRST_HS_INTENT', 'FIRST_CONTACT_CHANNEL', 'ANS_CR_VAL', 'ANS_EXP_VAL',
                               'ANS_NP_VAL', 'ANS_TT_CARES_VAL', 'ANS_CONF_CORR_VAL', 'ANS_EASY_RESOLVE_VAL',
                               'COBROWSE_FLAG', 'FIRST_SEARCH_MANUAL_LOCATION_DETAIL', 'SKILL_ID', 'CHANNEL_CARE',
                               'SRS_CATEGORY', 'WORKGROUP', 'WAIT_TIME']


    return care_cont_features, care_bool_features, care_catag_features, care_cont_score_features, care_bool_score_features, care_catag_score_features
