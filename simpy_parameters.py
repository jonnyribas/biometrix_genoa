<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 04:01:24 2017

@author: Afonso
"""



class Parameters:
    # Constants and global variables
    # just default values for testing
    OPEN = True
    ACQ_MEAN = 1.0
    SAMPLES_NUMBER = 100
    NUM_STATIONS = 1
    DAYS = 365
    RANDOM_SEED = 1000
    PT_MEAN = 1.0
    PT_SIGMA = 0.1
    MATCH_MEAN = 0.0
    CALL_MEAN = 1.0
    JOB_DURATION = 0.0
    EXPERT_DAY_OFF = 0
    EXAMINER_DAY_OFF = 0
    NUM_EXAMINERS = 10
    COST_EXAMINER = 1
    SHIFT_EXAMINER = 24
    NUM_EXPERTS = 10
    COST_EXPERT = 1
    SHIFT_EXPERT = 24
    SIM_TIME = 1
    SCORE_MIN = -10
    SCORE_MAX = 10
    SAMPLES_NUMBER = 100
    NUM_STATIONS = 1                                    # Number of acquisition stations
    DAYS = [1,2,3,4,5,6,7]                              # Working days
    EXPERT_DAY_OFF = []
    EXAMINER_DAY_OFF = []
    SHIFT_EXPERT_ST = 0
    SHIFT_EXAMINER_ST = 0
    SHIFT_TRIAGE = 24
    SHIFT_TRIAGE_ST = 5
    TRIAGE_DAY_OFF = 2
    NUM_TRIAGE = 1
    NUM_DAYS = 1
    NUM_REP = 1
    MEAN_ESCALATION_RATE = 1.0
    
    FMT_TIME = (1, 0.1)
    TRIAGE_TIME = (1, 0.1)
    FIU_TIME = (1, 0.1)
    
    REFERRAL_SIM = ['AR', 'FTA']
    PROCEDURE_TYPES = ["lights_out_match", "possible_match", "no_match"]
    REFERRAL_TYPES = ['AR', 'CR', 'FTA', 'CF', 'OTR']
    FMT_TYPES = ['1to1', '1toWL', '1toM']
    REFERRAL_DEST = FMT_TYPES + ['free']
    REFERRAL_PROB = [57, 17, 5, 12, 9]
    REFERRAL_PERC = {	 'AR': [0.069, 0.0214, 0.013, 0.8966],
                        'CR': [0.0, 0.2153, 0.1111, 0.6736],
                        'FTA': [0.0, 0.0214, 0.0, 0.9786],
                        'CF': [0.0, 0.2153, 0.111, 0.6736],
                        'OTR': [0.069, 0.0214, 0.013, 0.8966]}
    FMT_PERC = dict()
    TRIAGE_PERC = dict()
    TRIA_REFERRAL = dict()
    IMPOSTOR_PERC = [0.10, 0.10, 0.10, 0.10, 0.10]
    MATCHING = dict()

class arrivalParam:
    DAYS = [1,2,3,4,5,6,7]              # day of week
    OPEN_HOURS = 8                      # number the hours/day with arrival OPEN
    OPEN_START =  0                     # start hour of the arrival process (0...24)
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 04:01:24 2017

@author: Afonso
"""



class Parameters:
    # Constants and global variables
    # just default values for testing
    OPEN = True
    ACQ_MEAN = 1.0
    SAMPLES_NUMBER = 100
    NUM_STATIONS = 1
    DAYS = 365
    RANDOM_SEED = 1000
    PT_MEAN = 1.0
    PT_SIGMA = 0.1
    MATCH_MEAN = 0.0
    CALL_MEAN = 1.0
    JOB_DURATION = 0.0
    EXPERT_DAY_OFF = 0
    EXAMINER_DAY_OFF = 0
    NUM_EXAMINERS = 10
    COST_EXAMINER = 1
    SHIFT_EXAMINER = 24
    NUM_EXPERTS = 10
    COST_EXPERT = 1
    SHIFT_EXPERT = 24
    SIM_TIME = 1
    SCORE_MIN = -10
    SCORE_MAX = 10
    SAMPLES_NUMBER = 100
    NUM_STATIONS = 1                                    # Number of acquisition stations
    DAYS = [1,2,3,4,5,6,7]                              # Working days
    EXPERT_DAY_OFF = []
    EXAMINER_DAY_OFF = []
    SHIFT_EXPERT_ST = 0
    SHIFT_EXAMINER_ST = 0
    SHIFT_TRIAGE = 24
    SHIFT_TRIAGE_ST = 5
    TRIAGE_DAY_OFF = 2
    NUM_TRIAGE = 1
    NUM_DAYS = 1
    NUM_REP = 1
    MEAN_ESCALATION_RATE = 1.0
    FMT_TIME = (1, 0.1)
    TRIAGE_TIME = (1, 0.1)
    REFERRAL_SIM = ['AR', 'FTA']
    PROCEDURE_TYPES = ["lights_out_match", "possible_match", "no_match"]
    REFERRAL_TYPES = ['AR', 'CR', 'FTA', 'CF', 'OTR']
    FMT_TYPES = ['1to1', '1toWL', '1toM']
    REFERRAL_DEST = FMT_TYPES + ['free']
    REFERRAL_PROB = [57, 17, 5, 12, 9]
    REFERRAL_PERC = {	 'AR': [0.069, 0.0214, 0.013, 0.8966],
                        'CR': [0.0, 0.2153, 0.1111, 0.6736],
                        'FTA': [0.0, 0.0214, 0.0, 0.9786],
                        'CF': [0.0, 0.2153, 0.111, 0.6736],
                        'OTR': [0.069, 0.0214, 0.013, 0.8966]}
    FMT_PERC = dict()
    TRIAGE_PERC = dict()
    TRIA_REFERRAL = dict()
    IMPOSTOR_PERC = [0.10, 0.10, 0.10, 0.10, 0.10]
    MATCHING = dict()

class arrivalParam:
    DAYS = [1,2,3,4,5,6,7]              # day of week
    OPEN_HOURS = 8                      # number the hours/day with arrival OPEN
    OPEN_START =  0                     # start hour of the arrival process (0...24)
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
    DAY_OFF = []                         # number of day's off - i.e.: weekends = 2