"""
Biometric Examination Office

Covers:

- Interrupts
- Resources: PreemptiveResource

Scenario:
  A biometric examination office has *n* examiners. A stream of samples (enough to
  keep the examiners busy) arrives. Each examiner needs help for further inspection
  periodically. Extra investigations are carried out by experts. The experts
  have other, less important tasks to perform, too. Examiners
  preempt theses tasks. The experts continue them when they are done
  with the further inspections. The biomteric examination office works continuously.

"""
import random
import numpy
import simpy
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from simpy_parameters import *

#from pympler import tracker
#tr = tracker.SummaryTracker()
"""
    logging module:  https://docs.python.org/3/library/logging.html
    chage level=logging.INFO to:

    DEBUG: shows debug only msgs
    INFO: shows info only msgs
    WARNING: shows warning only msgs
    ERROR: shows error only msgs
    CRITICAL: shows critical only msgs

"""
import logging
from imp import reload
reload(logging)
#logging.basicConfig(format='%(message)s', level=logging.DEBUG)
#logging.basicConfig(format='%(message)s', filename='logging\simulation.log', filemode='w', level=logging.INFO)

from itertools import accumulate
from bisect import bisect

# input parameters
# Referral percentage for 'AR', 'CR', 'FTA', 'CF', 'OTR'
REFERRAL_PROB = [57, 17, 5, 12, 9]

# Referral percentage 1to1, 1toWL, 1toM and (1-1to1-1toWL-1toM)
triageWeightDict = {'AR': [0.069, 0.0214, 0.013, 0.8966],
                    'CR': [0.0, 0.2153, 0.1111, 0.6736],
                    'FTA': [0.0, 0.0214, 0.0, 0.9786],
                    'CF': [0.0, 0.2153, 0.111, 0.6736],
                    'OTR': [0.069, 0.0214, 0.013, 0.8966]}

# Operator referral percentage 1to1, 1toWL, 1toM
fmtWeigthDict = {'AR': dict(zip(FMT_TYPES, [0.0033, 0.0035, 0.0262])),
                 'CR': dict(zip(FMT_TYPES,[0.0, 0.0035, 0.0197])),
                 'FTA': dict(zip(FMT_TYPES,[0.0, 0.0035, 0.0])),
                 'CF': dict(zip(FMT_TYPES,[0.0, 0.0035, 0.0197])),
                 'OTR': dict(zip(FMT_TYPES,[0.0033, 0.0035, 0.0262]))}

# Impostors Percentage for 'AR', 'CR', 'FTA', 'CF', 'OTR'
impostorProb = [0.10, 0.10, 0.10, 0.10, 0.10]

impostorProbDict = dict(zip(REFERRAL_TYPES,impostorProb))


# pandas output
def createDF(index, columns, dat):
    dat = (dat + ', ')*(len(columns)-1)+dat
    values = numpy.zeros(len(index), dtype=dat)
    values.dtype.names = columns
    return pd.DataFrame(values, index=index, columns=columns)

def incDataframe(df, index, column, value=1):
    df.set_value(index, column, (df.loc[index, column]+value))

#############################################################################

def choicesDist(choices, weights):
    # distribution for referral type
    cumdist = list(accumulate(weights))
    x = random.random() * cumdist[-1]
    return choices[bisect(cumdist, x)]

def distributions(distType, ref=None):
    # all distribution in this model
    return {
        # Return actual processing time for examination of a biometric sample.
        'time_per_examination': random.normalvariate(PT_MEAN, PT_SIGMA),
        # Return actual inspection time for T1.
        't1_time_per_inspection': random.normalvariate(T1_INSPECTION_MEAN, T1_INSPECTION_SIGMA),
        # Return actual inspection time for T2.
        't2_time_per_inspection': random.normalvariate(T2_INSPECTION_MEAN, T2_INSPECTION_SIGMA),
        # Return time per examination at FMT
        'fmt_processing_time': random.normalvariate(*FMT_TIME),
        # Return escalation time for experts.
        'time_per_escalation': random.expovariate(CALL_MEAN),
        # Return actual acquisition time for sample.
        'time_per_acquisition': random.expovariate(ACQ_MEAN),
        # Return random scores for the samples.
        'score_per_sample': random.normalvariate(0.5, 0.5),
        # Return biometric system match time.
        'match_time': MATCH_MEAN,
        # Return the referral type
        'referral_type': choicesDist(REFERRAL_TYPES, REFERRAL_PROB)
    }.get(distType, 0.0)

def queueAvg(st):
    # return the mean of the list queue
    s_w = 0.0
    s_wx = 0.0
    for i in range(len(st)-1):
        s_w += (st[i+1][0]-st[i][0])
        s_wx += (st[i+1][0]-st[i][0])*(st[i][1])
    return s_wx / s_w if s_w > 0 else 0.0

def queuePlot(st, plt, shift, labels):
    # plot the list queue
    t=[]
    y=[]
    for i in range(len(st)):
        t.append(st[i][0]/(60.0*24.0))
        y.append(st[i-1][1])
    plt.step(t, y, label=labels)
    plt.legend(loc='best')
    numShifts = int(SIM_TIME/max(shift[1]-shift[0], 24*60))
    step = 1
    if SIM_TIME > 15*24*60:
        step = 7
    for i in range(numShifts):
        plt.axvspan(shift[0]/(60.0*24.0) + i, shift[1]/(60.0*24.0) + i, facecolor='b', alpha=0.1)
    plt.set_xticks(numpy.arange(min(t), max(t)+1, step))

#def confidenceInterval(a, conf=0.95):
#    # return average and confidence interval for values in a
#    avg, sem, m = numpy.mean(a), scipy.stats.sem(a), scipy.stats.t.ppf((1+conf)/2., len(a)-1)
#    h = m*sem
#
#    return avg, h

#############################################################################

class Staff(object):
    def __init__(self, env, shiftLenght, shiftStart, daysOff, numStaff):
        self.env = env
        self.numStaff = numStaff
        self.staffRes = simpy.Resource(env, capacity=numStaff)

        self.samples_inspected = 0                      # samples inspected by staff
        self.samples_resolved = 0                       # samples acepted + reject at tier
        self.samples_day_inspected = 0                  # number the samples inspect by staff/day
        self.timeInQueueT1 = 0.0                        # total time in tier 1 queue
        self.timeInQueueT2 = 0.0                        # total time in tier 2 queue
        self.acqQueueT1 = 0                             # samples acquiried from T1 queue
        self.acqQueueT2 = 0                             # samples acquiried from T2 queue
        self.busyTime = 0.0                             # total busy staff time
        self.appTimeResolved = 0.0                      # total sample time from start to resolved
        self.open = False                               # if shift is open or closed
        self.nextShift = 0.0                            # env.now of the next shift open/close
        self.time2op = 0.0                              # disp time for operation

        env.process(self.shift(shiftStart, shiftLenght, daysOff))             # shift process

    def shift(self, shiftStart, shiftLenght, daysOff):
        # shift control
        global inQueueT1, inQueueT2

        requestAll = yield self.env.process(self.getAllRes())
        get = True
        yield self.env.timeout(shiftStart)

        while True:
            for day in DAYS:
                self.samples_day_inspected = 0
                if day not in daysOff:
                    # working day
                    if get:
                        self.open = True
                        yield self.env.process(self.putAllRes(requestAll))
                        get = False

                    logging.debug(' %4.1f %s \tSHIFT ON\t%s' %(self.env.now, self.name, day))
                    start = self.env.now
                    yield self.env.timeout(shiftLenght)
                    self.open = False
                    self.nextShift = self.env.now + 24*60 - shiftLenght
                    logging.debug(' %4.1f %s \tSHIFT OFF\tSample inspect during shift: %i\t%s' %(self.env.now, self.name, self.samples_day_inspected, day))
                    # get all staff
                    x = self.name
                    y = self.staffRes.count
                    requestAll = yield self.env.process(self.getAllRes())
                    self.time2op += self.env.now - start
                    logging.debug(' %4.1f %s \tGET ALL\t%i' %(self.env.now, self.name, self.staffRes.count))
                    get = True
                    yield self.env.timeout(max(self.nextShift - self.env.now,0))

                else:
                    # weekend
                    self.open = False
                    logging.debug(' %4.1f %s\tSHIFT OFF WEEKEND\t%s' %(self.env.now, self.name, day))
                    self.nextShift = self.env.now + 24*60
                    if not get:
                        requestAll = yield self.env.process(self.getAllRes())
                        get = True

                    yield self.env.timeout(max(self.nextShift - self.env.now,0))

                # append the actual backlog
                backlogT1List.append(inQueueT1)
                backlogT2List.append(inQueueT2)

    def getAllRes(self):
    # request all resources during shift
        requestAll = [self.staffRes.request() for staff in range(self.numStaff)]
        yield self.env.all_of(requestAll)
        logging.debug(' %4.1f %s \tGET ALL\t%i' %(self.env.now, self.name, self.staffRes.count))
        self.env.exit(requestAll)

    def putAllRes(self, requestAll):
        for req in requestAll:
            yield self.staffRes.release(req)
        logging.debug(' %4.1f %s \tPUT ALL\t%i' %(self.env.now, self.name, self.staffRes.count))

#############################################################################

class Examiner(Staff):
    """An examiner evaluate biometric samples and may need help every now and then.

    If it cannot proceed, it calls an *experts* and continues the examination
    after further inspection is finalized.

    An examiner has a *name* and a number of *samples_inspected* thus far.

    """
    def __init__(self, env, rate):
        Staff.__init__(self, env, SHIFT_EXAMINER, SHIFT_EXAMINER_ST, EXAMINER_DAY_OFF, NUM_EXAMINERS)
        self.name = 'EXAMINER'
        self.expertRate = rate

        self.samplesSendToExperts = 0

    def callFMT(self, sample, experts):
        global inQueueT1, peak_resolved_time
        """Examines samples as long as the simulation runs.

        While examining a sample, the examiner may need help multiple times.
        Request an experts when this happens.

        """
        with self.staffRes.request() as req:
            logging.debug(' %4.1f %s  \tREQUEST\t\t%s' %(self.env.now, self.name, sample.name))
            yield req
            logging.debug(' %4.1f %s  \tGET EXAMI\t%s' %(self.env.now, self.name, sample.name))

            # remove from queue T1
            inQueueT1 -= 1
            self.timeInQueueT1 += self.env.now - sample.enterT1
            self.acqQueueT1 += 1
            queueT1List.append((self.env.now, inQueueT1))
            examination_time = numpy.abs(distributions('fmt_processing_time'))
            logging.debug(' %4.1f %s  \tST_EXAM\t\t%s\t%4.1f' %(self.env.now, self.name, sample.name, examination_time))
            yield self.env.timeout(examination_time)
            atTimeTotal[sample.referral] += examination_time

            self.samples_day_inspected += 1
            self.samples_inspected += 1
            self.busyTime += examination_time
            logging.debug(' %4.1f %s  \tEND_EXAM\t%s\tSamples examined @T1: %i'
                        %(self.env.now, self.name, sample.name, self.samples_inspected))

            # send to Expert?
            if random.random() <= fmtWeigthDict[sample.referral][sample.triage]:
                # send to expert.
                logging.debug(' %4.1f %s  \tTO FIU.\t\t%s\tAverage time from start to T1: %3.2f min'
                            %(self.env.now, self.name, sample.name, self.appTimeResolved/self.samples_resolved))
                self.env.process(self.sent2FIU(sample, experts))

            else:
                # inspection is done.
                self.samples_resolved += 1
                appTime = self.env.now - sample.start
                self.appTimeResolved += appTime
                resolvedTimeList.append(appTime)
                if peak_resolved_time < appTime:
                    peak_resolved_time = appTime
                logging.debug(' %4.1f %s  \tSOLVED.\t\t%s\tAverage time from start to T1: %3.2f min'
                            %(self.env.now, self.name, sample.name, self.appTimeResolved/self.samples_resolved))
            # release the staff resource

            logging.debug(' %4.1f %s  \tRELEASE\t\t%s' %(self.env.now, self.name, sample.name))


    def sent2FIU(self, sample, experts):
        global inQueueT2

        # send to Tier 2 queue procedure
        while not self.open:
            # wait until next shift opens
            yield self.env.timeout(self.nextShift - self.env.now)

        # increment No Referral count
        incDataframe(df_referralRate, 'Referral to FIU', sample.referral)
        incDataframe(df_toFIU, sample.referral, sample.triage)
        incDataframe(df_impFIU, sample.referral, sample.triage)

        self.samplesSendToExperts += 1
        logging.debug(' %4.1f %s  \tSEND T2\t\t%s\tTotal send to Tier 2: %i\tNumber in queue 2: %i'
                    %(self.env.now, self.name, sample.name, self.samplesSendToExperts, inQueueT2))
        inQueueT2 += 1
        sample.type = 'T2'
        sample.enterT2 = self.env.now
        queueT2List.append((self.env.now, inQueueT2))
        self.env.process(experts.operation(sample))

#############################################################################

class Expert(Staff):
    def __init__(self, env):
        Staff.__init__(self, env, SHIFT_EXPERT, SHIFT_EXPERT_ST, EXPERT_DAY_OFF, NUM_EXPERTS)
        self.name = 'EXPERT  '
        self.samples_examined = 0               # samples inspected from T1 queue
        self.samples_resolved

    def operation(self, sample):
        global inQueueT2, peak_resolved_time

        with self.staffRes.request() as req:
            yield req
            # remove from queue T2
            logging.debug(' %4.1f %s  \tGET FROM T2\t%s\tinQueueT2: %i\ttype:%s'
                %(self.env.now, self.name, sample.name, inQueueT2, sample.type))
            inQueueT2 -= 1
            self.timeInQueueT2 += self.env.now - sample.enterT2
            self.acqQueueT2 += 1
            queueT2List.append((self.env.now, inQueueT2))

            examination_time = numpy.abs(distributions('fmt_processing_time'))
            yield self.env.timeout(examination_time)
            opTimeTotal[sample.referral] += examination_time

            self.samples_inspected += 1
            self.samples_day_inspected += 1
            self.busyTime += examination_time
            self.samples_resolved += 1
            appTime = self.env.now - sample.start
            self.appTimeResolved += appTime
            resolvedTimeList.append(appTime)
            if peak_resolved_time < appTime:
                peak_resolved_time = appTime

            logging.debug(' %4.1f %s  \tEND EXAM\t%s\tSamples examined @T2: %i'
                                %(self.env.now, self.name, sample.name, self.samples_inspected))

#############################################################################

class Sample(object):
    def __init__(self, name, referral=None, triage=None, impostor=False, sample_type='T1', start=0.0, enterT1=None, enterT2=None):
        self.name = name
        self.type = sample_type
        self.start = start
        self.referral = referral
        self.triage = triage
        self.enterT1 = enterT1
        self.enterT2 = enterT2
        self.impostor = impostor

class Station(object):
    def __init__(self, env, name, examiners, experts):
        self.env = env
        self.name = name.upper()
        self.bioEngine = simpy.Resource(env, capacity=1)
        #self.examiners = examiners
        #self.experts = experts

        self.referralType = ''                  # referral type AR, CR, FTA, CF, OTR
        self.triageType = ''                    # 1to1, 1toWL or 1toM
        self.samples_created = 0                # samples created
        self.samples_acquired = 0               # samples acquired by biometric
        self.samplesRejected = 0                # samples reject by biometric
        self.samplesMatched = 0                 # samples acepted by biometric
        self.samples_passed = 0                 # samples acepted+rejected by biometric
        self.samples_queued = 0                 # samples queued to T1 by biometric
        self.acqQueueTime = 0.0                 # acquisition queue time
        self.bioEngBusyTime = 0.0               # biometric Engine busy time
        self.timeInStation = 0.0                # total sample time at Station (queue + biometric engine)
        self.impostor = False                   # impostor or not

        env.process(self.createSample(examiners, experts))

    def createSample(self, examiners, experts):
        """Create samples process"""
        while True:
            acquisition_time = numpy.abs(distributions('time_per_acquisition'))
            yield self.env.timeout(acquisition_time)
            self.samples_created += 1
            self.referralType = distributions('referral_type')
            populationTotal[self.referralType] += 1

            self.env.process(self.station_call('Sample %i' % self.samples_created, self.samples_created, examiners, experts))

    def station_call(self, sample, numSample, examiners, experts):
        """Match analysis at Biometric Engine"""
        global inQueueT1

        logging.debug(' %4.1f %s\tNEW ARRIV\t%s\tAcquisition queue size: %i'
                    %(self.env.now, self.name,sample, len(self.bioEngine.queue)))
        start = self.env.now
        # get bioEngine resource to match analyis
        with self.bioEngine.request() as req:
            yield req
            self.acqQueueTime += self.env.now - start
            logging.debug(' %4.1f %s\tST_MATCH\t%s' %(self.env.now, self.name, sample))
            # start match analysis
            enterBioEng = self.env.now
            yield self.env.timeout(distributions('match_time'))
            self.samples_acquired += 1
            self.bioEngBusyTime += self.env.now - enterBioEng
            self.triageType = choicesDist(TRIAGE_TYPES, triageWeightDict[self.referralType])
            if random.random() <= impostorProbDict[self.referralType]:
                # we have a impostor!
                self.impostor = True
                impostorDict[self.referralType] += 1

            #sample_score = distributions('score_per_sample')
            logging.debug(' %4.1f %s\tEND_MATCH\t%s\t\tTriage: %s' %(self.env.now, self.name, self.referralType, self.triageType))

        self.timeInStation += self.env.now - start

        if not self.triageType == 'free':
            # increment No Referral count
            incDataframe(df_referralRate, 'Referral to FMT', self.referralType)
            incDataframe(df_toFMT, self.referralType, self.triageType)
            incDataframe(df_impFMT, self.referralType, self.triageType)

            self.samples_queued += 1
            inQueueT1 += 1
            queueT1List.append((self.env.now, inQueueT1))
            self.env.process(examiners.callFMT(Sample(sample, self.referralType, self.triageType, impostor=self.impostor, start=start, enterT1=self.env.now), experts))
            logging.debug(' %4.1f %s\tFURTHER_I\t%s\tAverage time: %3.2f min'
                        %(self.env.now, self.name, sample, self.timeInStation/self.samples_acquired))
            self.instantStats()

        else:
            # increment No Referral count
            incDataframe(df_referralRate, 'No Referral', self.referralType)

            self.samplesMatched += 1
            self.samples_passed += 1
            logging.debug(' %4.1f %s\tMATCHED\t\t%s\tAverage time: %3.2f min'
                        %(self.env.now, self.name, sample, self.timeInStation/self.samples_acquired))
            self.instantStats()

    def instantStats(self):
        logging.debug(' %4.1f %s\tSTATS  \t\tArrived: %i\tAcquiried: %i\tPassed: %i\tFurther inv: %i.'
        %(self.env.now, self.name, self.samples_created, self.samples_acquired, self.samples_passed, self.samples_queued))

#############################################################################
def read_settings(settings_file):
    with open(settings_file) as json_data:
        #print (json_data)
        import_settings = json.load(json_data)
        json_data.close()
    return import_settings


def export_results(csv_file,data,fieldnames):

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(data)

#############################################################################
def getParameters():
    
    ############################################################################
    print('\nAssigning the parameters ...')

    global PT_MEAN, PT_SIGMA, MATCH_MEAN
    global CALL_MEAN, JOB_DURATION, EXPERT_DAY_OFF, EXAMINER_DAY_OFF
    global T1_INSPECTION_MEAN, T1_INSPECTION_SIGMA
    global T2_INSPECTION_MEAN, T2_INSPECTION_SIGMA
    global NUM_EXAMINERS, COST_EXAMINER, SHIFT_EXAMINER
    global NUM_EXPERTS, COST_EXPERT, SHIFT_EXPERT
    global SIM_TIME, SCORE_MIN, SCORE_MAX
    global ACQ_MEAN, NUM_STATIONS, DAYS, NUM_REP, RANDOM_SEED, NUM_DAYS
    global SHIFT_EXPERT_ST, SHIFT_EXAMINER_ST, MEAN_ESCALATION_RATE                                            # control T2 events

    #### impostor
    global FMT_TIME
    
    global processing_time, T1_inspection_time, T2_inspection_time

   # get all simulation parameters from json file
    logging.debug('Biometric Examination Office \nCost-Staff Statistical Analysis\n')
    logging.debug('Reading of settings ...')

    # get input parameters from jason file
    import_settings=read_settings("Parameters.json")
    # staffing
    full_time_equivalent_hours = int(import_settings["staffing_checking"]["full_time_equivalent_hours"])
    processing_time_seconds = int(import_settings["staffing_checking"]["processing_time_seconds"])
    unlink_investigation_time_seconds = int(import_settings["staffing_checking"]["unlink_investigation_time_seconds"])
    unlink_record_processing_seconds = int(import_settings["staffing_checking"]["unlink_record_processing_seconds"])
    link_record_processing_seconds = int(import_settings["staffing_checking"]["link_record_processing_seconds"])
    average_staff_absence_percentage = int(import_settings["staffing_checking"]["average_staff_absence_percentage"])

    # watch_list
    full_time_equivalent_hours = float(import_settings["watch_list"]["imposter_rate_percent"])
    adult_enteries = int(import_settings["watch_list"]["adult_enteries"])
    adult_expected_application = int(import_settings["watch_list"]["adult_expected_application"])
    adult_biographics_pick = int(import_settings["watch_list"]["adult_biographics_pick"])
    adult_no_biographics_pick = int(import_settings["watch_list"]["adult_no_biographics_pick"])
    adult_manual_error_percent = int(import_settings["watch_list"]["adult_manual_error_percent"])
    child_enteries = int(import_settings["watch_list"]["child_enteries"])
    child_expected_application = int(import_settings["watch_list"]["child_expected_application"])
    child_biographics_pick = int(import_settings["watch_list"]["child_biographics_pick"])
    child_no_biographics_pick = int(import_settings["watch_list"]["child_no_biographics_pick"])
    child_manual_error_percent = int(import_settings["watch_list"]["child_manual_error_percent"])

    random_seed = int(import_settings["simulation_settings"]["random_seed"])
    biometric_system_match_time = int(import_settings["simulation_settings"]["biometric_system_match_time"])
    examiners_day_off = int(import_settings["simulation_settings"]["examiners_day_off"])
    experts_day_off = int(import_settings["simulation_settings"]["experts_day_off"])

    shif_examiner_start = int(import_settings["simulation_settings"]["shif_examiner_start"])
    shif_expert_start = int(import_settings["simulation_settings"]["shif_expert_start"])

    mean_acquisition_rate = float(import_settings["simulation_settings"]["mean_acquisition_rate"])
    examiner_cost = float(import_settings["simulation_settings"]["examiner_cost"])
    examiners_number = int(import_settings["simulation_settings"]["examiners_number"])
    examiners_shift = float(import_settings["simulation_settings"]["examiners_shift"])
    expert_cost = float(import_settings["simulation_settings"]["expert_cost"])
    experts_number = int(import_settings["simulation_settings"]["experts_number"])
    experts_shift = float(import_settings["simulation_settings"]["experts_shift"])
    t1_inspection_time_min = int(import_settings["simulation_settings"]["t1_inspection_time_min"])
    t1_inspection_time_max = int(import_settings["simulation_settings"]["t1_inspection_time_max"])
    t2_inspection_time_min = int(import_settings["simulation_settings"]["t2_inspection_time_min"])
    t2_inspection_time_max = int(import_settings["simulation_settings"]["t2_inspection_time_max"])
    mean_escalation_rate = float(import_settings["simulation_settings"]["mean_scalation_rate"])
    processing_time_min = int(import_settings["simulation_settings"]["processing_time_min"])
    processing_time_max = int(import_settings["simulation_settings"]["processing_time_max"])
    score_threshold_min = float(import_settings["simulation_settings"]["score_threshold_min"])
    score_threshold_max = float(import_settings["simulation_settings"]["score_threshold_max"])
    simulation_time = float(import_settings["simulation_settings"]["simulation_time"])
    number_of_replications = int(import_settings["simulation_settings"]["number_of_replications"])

    processing_time = [processing_time_min, processing_time_max]
    T1_inspection_time = [t1_inspection_time_min, t1_inspection_time_max]
    T2_inspection_time = [t2_inspection_time_min, t2_inspection_time_max]

    RANDOM_SEED = random_seed
    NUM_STATIONS = 1                                                # Number of acquisition stations
    MATCH_MEAN = biometric_system_match_time/30                     # Biometric system match time
    PT_MEAN = numpy.mean(processing_time)                           # Average of processing time in minutes
    PT_SIGMA = numpy.std(processing_time)                           # Deviation of processing time
    T1_INSPECTION_MEAN = numpy.mean(T1_inspection_time)             # Tier 1 Average inspection time in minutes
    T1_INSPECTION_SIGMA = numpy.std(T1_inspection_time)             # Tier 2 Deviation of inspection time in minutes
    T2_INSPECTION_MEAN = numpy.mean(T2_inspection_time)             # Tier 1 Average inspection time in minutes
    T2_INSPECTION_SIGMA = numpy.std(T2_inspection_time)             # Tier 2 Deviation of inspection time in minutes
    NUM_EXAMINERS = examiners_number                                # Number of examiners in the office
    SHIFT_EXAMINER = examiners_shift  * 60                          # Length of examiners' shift in minutes
    COST_EXAMINER = examiner_cost/60                                # cost of an examiner per hour
    NUM_EXPERTS = experts_number                                    # Number of experts in the office
    SHIFT_EXPERT = experts_shift * 60                               # Length of experts' shift in minutes
    COST_EXPERT = expert_cost/60                                    # cost of an expert per hour
    SIM_TIME = simulation_time * 24 * 60                            # Simulation time in minutes
    NUM_DAYS = simulation_time
    SCORE_MIN = score_threshold_min                                 # Minimum threshold
    SCORE_MAX = score_threshold_max                                 # Maximum threshold
    ACQ_MEAN = 1.0 / (24*60/mean_acquisition_rate)                  # Mean acquisition rates (samples per day) for exponential distribution
    CALL_MEAN = mean_escalation_rate * ACQ_MEAN                     # Mean time to call in minutes for exponential distribution
    SHIFT_EXAMINER_ST = shif_examiner_start*60                      # hour of shift start for examiners
    SHIFT_EXPERT_ST = shif_expert_start*60                          # hour of shift start for experts
    DAYS = [1,2,3,4,5,6,7]                                          # Working days
    MEAN_ESCALATION_RATE = mean_escalation_rate
    NUM_REP = number_of_replications                                # Number of replication
    FMT_MEAN_TIME = processing_time_seconds/60.0                    # FMT processing time in minutes
    FMT_TIME = (FMT_MEAN_TIME, 0.1*FMT_MEAN_TIME)                   # FMT distribution time parameters

    EXPERT_DAY_OFF = [i for i in range((len(DAYS) - experts_day_off + 1), len(DAYS) +1)]
    EXAMINER_DAY_OFF = [i for i in range((len(DAYS) - examiners_day_off + 1), len(DAYS) +1)]
     
# Setup and start the simulation

def simulate():
    
    global inQueueT1, inQueueT2, queueT1List, queueT2List
    global backlogT1List, backlogT2List, peak_resolved_time, resolvedTimeList
    
    peak_resolved_time = 0.0                                        # peak time taken for a sample to resolved
    resolvedTimeList = []                                           # list for time from acq to resolved
    inQueueT1, inQueueT2 = 0, 0                                     # number in queue for Tiers
    queueT1List, queueT2List = [(0,0)], [(0,0)]                     # global list to store number in queue over time
    backlogT1List, backlogT2List = [], []                           # global list to store backlog @shift begin



    #############################################################################
    #print('\nProcessing (%s-day period) ...' % simulation_time)
    logging.info(' Input parameters ...')
    logging.info(' Number of acquisition stations: %i'% NUM_STATIONS)
    logging.info(' Average of processing time in minutes: %3.2f', PT_MEAN)
    logging.info(' Deviation of processing time: %3.2f', PT_SIGMA)
    logging.info(' Average inspection time in minutes: %3.2f', T1_INSPECTION_MEAN)
    logging.info(' Deviation of inspection time in minutes: %3.2f', T1_INSPECTION_SIGMA)
    logging.info(' Number of examiners in the office: %i', NUM_EXAMINERS)
    logging.info(' Length of examiners shift in minutes: %3.2f', SHIFT_EXAMINER)
    logging.info(' Cost of an examiner per hour: %3.2f $/h', COST_EXAMINER)
    logging.info(' Number of experts in the office: %3.2f', NUM_EXPERTS)
    logging.info(' Length of experts shift in minutes: %3.2f', SHIFT_EXPERT)
    logging.info(' Cost of an expert per hour: %3.2f', COST_EXPERT)
    logging.info(' Simulation time in minutes: %3.2f', SIM_TIME)
    logging.info(' Minimum threshold: %3.2f', SCORE_MIN)
    logging.info(' Maximum threshold: %3.2f', SCORE_MAX)
    logging.info(' Mean acquisition rates for exponential distribution: %3.2f samples/min' %(ACQ_MEAN))
    #logging.info(' Mean time to call in minutes for exponential distribution: %3.2f', CALL_MEAN)
    #logging.info(' Probability to call Tier 2 from Tier 1: %3.2f', mean_escalation_rate)
    logging.info(' Daily enrolment volume with facial biometric: %3.2f samples/day', ACQ_MEAN/(24*60))
    logging.info(' Biometric system match time: %3.2f seg', MATCH_MEAN*30)

    # Create an environment
    env = simpy.Environment()

    # Start the setup process
    experts = Expert(env)
    examiners = Examiner(env, MEAN_ESCALATION_RATE)
    stations = [Station(env, 'Station %02d' % i, examiners, experts) for i in range(1,NUM_STATIONS+1)]

    # Execute!
    logging.info(' %4.1f Processing (%s-day period) ...' %(env.now, SIM_TIME/(24*60)))
    # Days of simulation

    env.run(until=SIM_TIME)

    #print('\nOverall statistics ...\n')
    logging.info('\nOverall statistics ...')
    examiners_samples = examiners.samples_inspected
    # appTimeResolved: time from sample create to be resolved (leave the system by T1 and/or T2)
    examiners_times = examiners.appTimeResolved
    examiners_cost = examiners.busyTime*COST_EXAMINER
    #examiners_rate = examiners.busyTime/examiners_samples if examiners_samples > 0 else 0.0
    examiners_busy = examiners.busyTime/NUM_EXAMINERS
    examiners_utilization = examiners_busy/examiners.time2op

    experts_samples = experts.samples_inspected + experts.samples_examined
    experts_inspected = experts.samples_inspected
    experts_times = experts.appTimeResolved
    experts_cost = experts.busyTime*COST_EXPERT
    #experts_rate = experts.busyTime/experts_samples if experts_samples > 0 else 0.0
    experts_busy = experts.busyTime/NUM_EXPERTS
    experts_utilization = experts_busy/experts.time2op

    # all samples examined from T1: examiners and experts
    examined_samples = examiners_samples + experts.samples_examined

    created_samples = numpy.sum(station.samples_created for station in stations)
    acquired_samples = numpy.sum(station.samples_acquired for station in stations)
    waiting_samples = numpy.sum(len(station.bioEngine.queue) for station in stations)
    passed_samples = numpy.sum(station.samples_passed for station in stations)
    queued_samples = numpy.sum(station.samples_queued for station in stations)

    biometric_queue_time = numpy.sum(station.acqQueueTime for station in stations)/acquired_samples
    biometric_utilization = numpy.sum(station.bioEngBusyTime for station in stations)/SIM_TIME/NUM_STATIONS
    biometric_queue_avg = ACQ_MEAN * biometric_queue_time  # little´s law

    avg_resolved_time = (examiners.appTimeResolved + experts.appTimeResolved)/(examiners.samples_resolved + experts.samples_resolved)

    total_times = numpy.sum([examiners_times,experts_times])
    total_costs = numpy.sum([examiners_cost,experts_cost])
    total_rates = (examiners.samples_resolved + experts.samples_resolved)/(SIM_TIME)
    #total_rates = numpy.sum([examiners_samples*examiners_rate,experts_samples*experts_rate])/numpy.sum([examiners_samples,experts_samples])

    queue_T1_avg_time = (examiners.timeInQueueT1 + experts.timeInQueueT1)/(examiners.acqQueueT1 + experts.acqQueueT1) if examiners.acqQueueT1 + experts.acqQueueT1>0 else 0.0
    queue_T2_avg_time = (experts.timeInQueueT2)/(experts.acqQueueT2) if experts.acqQueueT2 > 0 else 0.0

    examiners_mean_backlog = 24 * 60 * (queued_samples - examined_samples)/SHIFT_EXAMINER
    #examiners_mean_backlog = 24 * 60 * (queued_samples - examiners_samples)/SHIFT_EXAMINER

    experts_mean_backlog = 24 * 60 * (examiners.samplesSendToExperts - experts_inspected)/SHIFT_EXPERT

    #examiners_mean_backlog = numpy.mean(backlogT1List)
    #experts_mean_backlog =  numpy.mean(backlogT2List)

    queueT1List.append((env.now, inQueueT1))
    queueT2List.append((env.now, inQueueT2))

    examiners_mean_queue = queueAvg(queueT1List)
    experts_mean_queue = queueAvg(queueT2List)

    queue_length_at_end = queued_samples - (examiners.samples_resolved + experts.samples_resolved)

    output = []

    fields = ["Sample_Acquisition_Rate",
              "T1_Inspection_Time_Min",
              "T1_Inspection_Time_Max",
              "T2_Inspection_Time_Min",
              "T2_Inspection_Time_Max",
              "Expert_Inspection_Cost",
              "Experts_Mean_Backlog",
              "Examiner_Processing_Time_Min",
              "Examiner_Processing_Time_Max",
              "Examiner_Processing_Cost",
              "Examiners_Mean_Backlog",
              "Threshold_Min",
              "Threshold_Max",
              "Experts_Number",
              "Examiners_Number",
              "Created_Samples_Number",
              "Acquired_Samples_Number",
              "Waiting_Biometric_Samples_Number",
              "Biometric_Utilization",
              "Biometric_Queue_Time",
              "Biometric_Queue_Avg",
              "Passed_Samples_Number",
              "Examined_Samples_Number",
              "Inspected_Samples_Number",
              "Queued_Samples_Number",
              "Total_Processing_Time",
              "Total_Processing_Rate",
              "Average_Resolved_Time",
              "Peak_Resolved_Time",
              "First_Quartile_Resolved_Time",
              "Second_Quartile_Resolved_Time",
              "Third_Quartile_Resolved_Time",
              "Queue_Length_At_End",
              "Queue_T1_Average_Time",
              "Queue_T2_Average_Time",
              "Queue_T1_Average_Samples",
              "Queue_T2_Average_Samples",
              "Examiners_Utilization",
              "Experts_Utilization",
              "Total_Cost"]

    output.append({"Sample_Acquisition_Rate": ACQ_MEAN/(24*60),
                   "T1_Inspection_Time_Min": T1_inspection_time[0],
                   "T1_Inspection_Time_Max": T1_inspection_time[1],
                   "T2_Inspection_Time_Min": T2_inspection_time[0],
                   "T2_Inspection_Time_Max": T2_inspection_time[1],
                   "Expert_Inspection_Cost": COST_EXPERT,
                   "Experts_Mean_Backlog": experts_mean_backlog,
                   "Examiner_Processing_Time_Min": processing_time[0],
                   "Examiner_Processing_Time_Max": processing_time[1],
                   "Examiner_Processing_Cost": COST_EXAMINER,
                   "Examiners_Mean_Backlog": examiners_mean_backlog,
                   "Threshold_Min": SCORE_MIN,
                   "Threshold_Max": SCORE_MAX,
                   "Experts_Number": NUM_EXPERTS,
                   "Examiners_Number": NUM_EXAMINERS,
                   "Created_Samples_Number": created_samples,
                   "Acquired_Samples_Number": acquired_samples,
                   "Waiting_Biometric_Samples_Number": waiting_samples,
                   "Biometric_Utilization": biometric_utilization,
                   "Biometric_Queue_Time": biometric_queue_time,
                   "Biometric_Queue_Avg": biometric_queue_avg,
                   "Passed_Samples_Number": passed_samples,
                   "Examined_Samples_Number": examined_samples,
                   "Inspected_Samples_Number": experts_inspected,
                   "Queued_Samples_Number": queued_samples,
                   "Total_Processing_Time": total_times/60,
                   "Total_Processing_Rate": total_rates,
                   "Average_Resolved_Time": avg_resolved_time,
                   "Peak_Resolved_Time": peak_resolved_time,
                   "First_Quartile_Resolved_Time": numpy.percentile(resolvedTimeList, 25),
                   "Second_Quartile_Resolved_Time": numpy.percentile(resolvedTimeList, 50),
                   "Third_Quartile_Resolved_Time": numpy.percentile(resolvedTimeList, 75),
                   "Queue_Length_At_End": queue_length_at_end,
                   "Queue_T1_Average_Time": queue_T1_avg_time,
                   "Queue_T2_Average_Time": queue_T2_avg_time,
                   "Queue_T1_Average_Samples": examiners_mean_queue,
                   "Queue_T2_Average_Samples": experts_mean_queue,
                   "Examiners_Utilization": examiners_utilization,
                   "Experts_Utilization": experts_utilization,
                   "Total_Cost": total_costs})

    logging.info(' Sample_Acquisition_Rate:  %4.1f samples/day' %( ACQ_MEAN/(24*60)))
    logging.info(' Acquisition_Rate:  %4.2f samples/day' %(queued_samples/ ACQ_MEAN/(24*60)))
    logging.info(' T1_Inspection_Time_Min:  %4.1f min' %(T1_inspection_time[0]))
    logging.info(' T1_Inspection_Time_Max:  %4.1f min' %(T1_inspection_time[1]))
    logging.info(' T2_Inspection_Time_Min:  %4.1f min' %(T2_inspection_time[0]))
    logging.info(' T2_Inspection_Time_Max:  %4.1f min' %(T2_inspection_time[1]))
    logging.info(' Expert_Inspection_Cost:  $%4.1f' %(COST_EXPERT))
    logging.info(' Experts_Mean_Backlog:  %4.1f samples' %(experts_mean_backlog))
    logging.info(' Examiner_Processing_Time_Min:  %4.1f min' %(processing_time[0]))
    logging.info(' Examiner_Processing_Time_Max:  %4.1f min' %(processing_time[1]))
    logging.info(' Examiner_Processing_Cost:  $%4.1f' %(COST_EXAMINER))
    logging.info(' Examiners_Mean_Backlog:  %4.1f samples' %(examiners_mean_backlog))
    logging.info(' Threshold_Min:  %4.2f' %(SCORE_MIN))
    logging.info(' Threshold_Max:  %4.2f' %(SCORE_MAX))
    logging.info(' Experts_Number:  %4.1f experts' %(NUM_EXPERTS))
    logging.info(' Examiners_Number:  %4.1f examiners' %(NUM_EXAMINERS))
    logging.info(' Created_Samples_Number:  %4.1f samples' %(created_samples))
    logging.info(' Acquired_Samples_Number:  %4.1f samples' %(acquired_samples))
    logging.info(' Waiting_Biometric_Samples_Number:  %4.1f samples' %(waiting_samples))
    logging.info(' Biometric_Utilization:  %4.2f' %(biometric_utilization))
    logging.info(' Biometric_Queue_Time:  %4.1f min/sample' %(biometric_queue_time))
    logging.info(' Biometric_Queue_Avg:  %4.1f samples' %(biometric_queue_avg))
    logging.info(' Passed_Samples_Number:  %i samples' %(passed_samples))
    logging.info(' Examined_Samples_Number:  %i samples' %(examined_samples))
    logging.info(' Inspected_Samples_Number:  %i samples' %(experts_samples))
    logging.info(' Queued_Samples_Number:  %i samples' %(queued_samples))
    logging.info(' Total_Processing_Time:  %4.1f h' %(total_times/60))
    logging.info(' Total_Processing_Rate:  %4.1f sample/day' %(total_rates))
    logging.info(' Average_Resolved_Time:  %4.1f min' %(avg_resolved_time))
    logging.info(' Peak_Resolved_Time:  %4.1f min' %(peak_resolved_time))
    logging.info(' First_Quartile_Resolved_Time:  %4.1f min' %(numpy.percentile(resolvedTimeList, 25)))
    logging.info(' Second_Quartile_Resolved_Time:  %4.1f min' %(numpy.percentile(resolvedTimeList, 50)))
    logging.info(' Third_Quartile_Resolved_Time:  %4.1f min' %(numpy.percentile(resolvedTimeList, 75)))
    logging.info(' Queue_Length_At_End:  %4.0f samples' %(queue_length_at_end))
    logging.info(' Queue_T1_Average_Time:  %4.2f min/sample' %(queue_T1_avg_time))
    logging.info(' Queue_T2_Average_Time:  %4.2f min/sample' %(queue_T2_avg_time))
    logging.info(' Queue_T1_Average_Samples:  %4.1f sample' %(examiners_mean_queue))
    logging.info(' Queue_T2_Average_Samples:  %4.1f samples' %(experts_mean_queue))
    logging.info(' Examiners_Utilization:  %4.2f' %(examiners_utilization))
    logging.info(' Experts_Utilization:  %4.2f' %(experts_utilization))
    logging.info(' Total_Cost: $%4.1f' %(total_costs))



    return fields, output, queueT1List, queueT2List


if __name__ == '__main__':

    def sumReplication(data, column):
        # sum dataframe column
        df_Tmp = pd.DataFrame([data])
        df_Tmp = df_Tmp[REFERRAL_TYPES]
        df_Tmp = df_Tmp.transpose()
        df_popSim[column] = df_popSim[column] + df_Tmp[df_Tmp.columns[0]]/NUM_DAYS

    df_popSim = createDF(REFERRAL_TYPES,
                         ['Total Population', 'Total Impostors', 'Type Percentage', 'Impostors Percentage', 'Total FMT Time (min)', 'Total FIU Time (min)'],
                         'int32')

    df_toFMTSim = createDF(REFERRAL_TYPES, FMT_TYPES, 'float32')
    df_toFIUSim = createDF(REFERRAL_TYPES, FMT_TYPES, 'float32')
    df_AtTimeSim = createDF(REFERRAL_TYPES, FMT_TYPES, 'float32')
    df_OpTimeSim = createDF(REFERRAL_TYPES, FMT_TYPES, 'float32')
    
    getParameters()
    random.seed(RANDOM_SEED)  # This helps reproducing the results

    plt.close('all')
    fig, axes = plt.subplots(ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()
    fig.suptitle("Biometric Examination Office, Cost-Staff Statistical Analysis")
    ax1.set_xlabel("Simulation time (days)")
    ax2.set_xlabel("Simulation time (days)")
    ax1.set_ylabel("Referrals in FMT queue")
    ax2.set_ylabel("Referrals in FIU queue")
    
    for rep in range(NUM_REP):
        print('Replication', rep)
        impostorDict = dict(zip(REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        populationTotal = dict(zip(REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        atTimeTotal = dict(zip(REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        opTimeTotal = dict(zip(REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        df_referralRate = createDF(['No Referral', 'Referral to FMT', 'Referral to FIU'], REFERRAL_TYPES, 'int32')
        df_toFMT = createDF(REFERRAL_TYPES, FMT_TYPES, 'int32')
        df_toFIU = createDF(REFERRAL_TYPES, FMT_TYPES, 'int32')
        df_impFMT = createDF(REFERRAL_TYPES, FMT_TYPES, 'int32')
        df_impFIU = createDF(REFERRAL_TYPES, FMT_TYPES, 'int32')

        fields, output, queueT1List, queueT2List = simulate()
        queuePlot(queueT1List, ax1, (SHIFT_EXAMINER_ST, (SHIFT_EXAMINER_ST + SHIFT_EXAMINER)), 'Replication ' + str(rep))
        queuePlot(queueT2List, ax2, (SHIFT_EXPERT_ST, (SHIFT_EXPERT_ST + SHIFT_EXPERT)), 'Replication ' + str(rep))
        
        sumReplication(populationTotal,'Total Population')
        sumReplication(populationTotal,'Total Impostors')
        sumReplication(populationTotal,'Total FMT Time (min)')
        sumReplication(populationTotal,'Total FIU Time (min)')

        df_toFMTSim = df_toFMTSim + df_toFMT/NUM_DAYS
        df_toFIUSim = df_toFIUSim + df_toFIU/NUM_DAYS
    
    plt.show()
    df_popSim = df_popSim/NUM_REP
    df_toFMTSim = df_toFMTSim/NUM_REP
    df_toFIUSim = df_toFIUSim/NUM_REP

    columnsNames = [i+'_FMT' for i in FMT_TYPES]
    df_toFMTSim.columns = columnsNames

    columnsNames = [i+'_FIU' for i in FMT_TYPES]
    df_toFIUSim.columns = columnsNames

    totalTmp = numpy.sum(df_popSim['Total Population'])
    df_popSim['Type Percentage'] = df_popSim['Total Population']/totalTmp

    df_popSim['Impostors Percentage'] = df_popSim['Total Impostors']/df_popSim['Total Population']

    writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

    df_popSim.to_excel(writer)
    df_toFMTSim.to_excel(writer, startcol=8)
    df_toFIUSim.to_excel(writer, startcol=13)

    writer.save()