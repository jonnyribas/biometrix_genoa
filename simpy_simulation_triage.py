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
from simpy_parameters import Parameters as param, arrivalParam as arrivalParam

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

impostorProbDict = {}
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
        'time_per_examination': random.normalvariate(param.PT_MEAN, param.PT_SIGMA),
<<<<<<< HEAD
        # Return actual inspection time for FMT.
        'fmt_processing_time': random.normalvariate(*param.FMT_TIME),
        # Return time per examination at Triage
        'triage_processing_time': random.normalvariate(*param.TRIAGE_TIME),
        # Return actual inspection time for FIU.
        'fiu_processing_time': random.normalvariate(*param.FMT_TIME),
=======
        # Return actual inspection time for T1.
        'fmt_processing_time': random.normalvariate(*param.FMT_TIME),
        # Return time per examination at FMT
        'triage_processing_time': random.normalvariate(*param.TRIAGE_TIME),
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
        # Return escalation time for experts.
        'time_per_escalation': random.expovariate(param.CALL_MEAN),
        # Return actual acquisition time for sample.
        'time_per_acquisition': random.expovariate(param.ACQ_MEAN),
        # Return random scores for the samples.
        'score_per_sample': random.normalvariate(0.5, 0.5),
        # Return biometric system match time.
        'match_time': param.MATCH_MEAN,
        # Return the referral type
        'referral_type': choicesDist(param.REFERRAL_TYPES, param.REFERRAL_PROB)
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
    numShifts = int(param.SIM_TIME/max(shift[1]-shift[0], 24*60))
    step = 1
    if param.SIM_TIME > 15*24*60:
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
        self.staffRes = simpy.PriorityResource(env, capacity=numStaff)

        self.samples_inspected = 0                      # samples inspected by staff
        self.samples_resolved = 0                       # samples acepted + reject at tier
        self.samples_day_inspected = 0                  # number the samples inspect by staff/day
        self.timeInQueueT1 = 0.0                        # total time in tier 1 queue
        self.timeInQueueTriage = 0.0                    # total time in tier Triage queue
        self.timeInQueueT2 = 0.0                        # total time in tier 2 queue
        self.acqQueueT1 = 0                             # samples acquiried from T1 queue
        self.acqQueueTriage = 0                         # samples acquiried from Triage queue
        self.acqQueueT2 = 0                             # samples acquiried from T2 queue
        self.busyTime = 0.0                             # total busy staff time
        self.appTimeResolved = 0.0                      # total sample time from start to resolved
        self.open = False                               # if shift is open or closed
        self.nextShift = 0.0                            # env.now of the next shift open/close
        self.time2op = 0.0                              # disp time for operation

        env.process(self.shift(shiftStart, shiftLenght, daysOff))             # shift process

    def shift(self, shiftStart, shiftLenght, daysOff):
        # shift control
        global inQueueT1, inQueueTriage, inQueueT2

        requestAll = yield self.env.process(self.getAllRes())
        get = True
        yield self.env.timeout(shiftStart)

        while True:
            for day in param.DAYS:
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
                backlogTriageList.append(inQueueTriage)
                backlogT2List.append(inQueueT2)

    def getAllRes(self):
    # request all resources during shift
        requestAll = [self.staffRes.request(priority=0) for staff in range(self.numStaff)]
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
    def __init__(self, env):
        Staff.__init__(self, env, param.SHIFT_EXAMINER, param.SHIFT_EXAMINER_ST, param.EXAMINER_DAY_OFF, param.NUM_EXAMINERS)
        self.name = 'EXAMINER'

        self.samplesSendToTriage = 0




    def callFMT(self, sample, triage, experts):
        global inQueueT1, peak_resolved_time
        """Examines samples as long as the simulation runs.

        While examining a sample, the examiner may need help multiple times.
        Request an experts when this happens.

        """
        with self.staffRes.request(priority=1) as req:
            logging.debug(' %4.1f %s  \tREQUEST\t\t%s' %(self.env.now, self.name, sample.name))
            yield req
            logging.debug(' %4.1f %s  \tGET EXAMI\t%s' %(self.env.now, self.name, sample.name))

            # remove from queue T1
            inQueueT1 -= 1
            self.timeInQueueT1 += self.env.now - sample.enterT1
            self.acqQueueT1 += 1
            queueT1List.append((self.env.now, inQueueT1))
            if sample.impostor:
                impostorDictFMT[sample.referralType] += 1

            examination_time = numpy.abs(distributions('fmt_processing_time'))
            logging.debug(' %4.1f %s  \tST_EXAM\t\t%s\t%4.1f' %(self.env.now, self.name, sample.name, examination_time))
            yield self.env.timeout(examination_time)
            atTimeTotal[sample.referralType] += examination_time

            self.samples_day_inspected += 1
            self.samples_inspected += 1
            self.busyTime += examination_time
            logging.debug(' %4.1f %s  \tEND_EXAM\t%s\tSamples examined @T1: %i'
                        %(self.env.now, self.name, sample.name, self.samples_inspected))

            self.env.process(self.exitFMTTransfer(sample, triage, experts))
            logging.debug(' %4.1f %s  \tRELEASE\t\t%s' %(self.env.now, self.name, sample.name))

    def exitFMTTransfer(self, sample, triage, experts):
        global peak_resolved_time

        if random.random() <= param.FMT_REFERRAL[sample.referralType][sample.triageType][sample.matchType]:
            sample.match = True
            while not self.open:
                # wait until next shift opens
                yield self.env.timeout(self.nextShift - self.env.now)

            if param.TRIA_TRANSFER[sample.referralType][sample.triageType][sample.matchType] == 1:
                if param.TRIA_REFERRAL[sample.referralType][sample.triageType][sample.matchType] == 1:
                    if param.FIU_TRANSFER[sample.referralType][sample.triageType][sample.matchType] == 1:
<<<<<<< HEAD
                        send2FIU(sample, experts, self.env.now, self.name)
                        self.env.process(experts.callFIU(sample))
                    else:
                        # blocked
                        blockedDictFMT[sample.referralType] += 1
                elif param.TRIA_REFERRAL[sample.referralType][sample.triageType][sample.matchType] > 0:
                    # send to triage.
                    self.samplesSendToTriage += 1
                    send2Triage(sample, experts, self.env.now, 'FMT')
                    logging.debug(' %4.1f %s  \tTO TRIAGE.\t\t%sn' % (self.env.now, sample.name, sample.name))
                    self.env.process(triage.callTriage(sample, experts))
                else:
                    # blocked
                     blockedDictFMT[sample.referralType] += 1
            else:
                # blocked
                blockedDictFMT[sample.referralType] += 1
        else:
            # passport production
            passportDictFMT[sample.referralType] += 1

def send2Triage(sample, experts, now, name):
=======
                        self.env.process(experts.operation(sample))
                elif param.TRIA_REFERRAL[sample.referralType][sample.triageType][sample.matchType] > 0:
                    # send to triage.
                    self.samplesSendToTriage += 1
                    sent2Triage(sample, triage, experts, self.env.now, 'FMT')
                    logging.debug(' %4.1f %s  \tTO TRIAGE.\t\t%sn' % (self.env.now, sample.name, sample.name))
                    self.env.process(triage.callTriage(sample, experts))

#        # inspection is done.
#        self.samples_resolved += 1
#        appTime = self.env.now - sample.start
#        self.appTimeResolved += appTime
#        resolvedTimeList.append(appTime)
#        if peak_resolved_time < appTime:
#            peak_resolved_time = appTime
#        logging.debug(' %4.1f %s  \tSOLVED.\t\t%s\tAverage time from start to Finish FMT: %3.2f min'
#                    %(self.env.now, self.name, sample.name, self.appTimeResolved/self.samples_resolved))


def sent2Triage(sample, triage, experts, now, name):
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
    global inQueueTriage
    # increment No Referral count
    incDataframe(df_referralRate, 'Referral to FIU', sample.referralType)
    incDataframe(df_toTriage, sample.referralType, sample.triageType)
    incDataframe(df_impTriage, sample.referralType, sample.triageType)

    logging.debug(' %4.1f %s  \tSEND Triage\t\t%s\tNumber in queue 2: %i'
                %(now, name, sample.name, inQueueTriage))
    inQueueTriage += 1
    sample.type = 'Triage'
    sample.enterTriage = now
    queueTriageList.append((now, inQueueTriage))


#############################################################################

class Triage(Staff):
    """Triage unit only sees cases that have been seen by the FMT and are
        being referred to the FIU”.
    """
    def __init__(self, env):
        Staff.__init__(self, env, param.SHIFT_TRIAGE, param.SHIFT_TRIAGE_ST, param.TRIAGE_DAY_OFF, param.NUM_TRIAGE)
        self.name = 'TRIAGE'

        self.samplesSendToFIU = 0

    def callTriage(self, sample, experts):
        global inQueueTriage, peak_resolved_time
        """Examines samples as long as the simulation runs.

        While examining a sample, the examiner may need help multiple times.
        Request an experts when this happens.

        """
        with self.staffRes.request(priority=1) as req:
            logging.debug(' %4.1f %s  \tREQUEST\t\t%s' %(self.env.now, self.name, sample.name))
            yield req
            logging.debug(' %4.1f %s  \tGET TRIAGE\t%s' %(self.env.now, self.name, sample.name))

            # remove from queue T1
            inQueueTriage -= 1
            self.timeInQueueTriage += self.env.now - sample.enterTriage
            self.acqQueueTriage += 1
            queueTriageList.append((self.env.now, inQueueTriage))
            if sample.impostor:
                impostorDictTriage[sample.referralType] += 1

            examination_time = numpy.abs(distributions('triage_processing_time'))
            logging.debug(' %4.1f %s  \tST_EXAM\t\t%s\t%4.1f' %(self.env.now, self.name, sample.name, examination_time))
            yield self.env.timeout(examination_time)
            triageTimeTotal[sample.referralType] += examination_time

            self.samples_day_inspected += 1
            self.samples_inspected += 1
            self.busyTime += examination_time
            logging.debug(' %4.1f %s  \tEND_EXAM\t%s\tSamples examined @Triage: %i'
                        %(self.env.now, self.name, sample.name, self.samples_inspected))
            self.env.process(self.exitTriageTransfer(sample, experts))

<<<<<<< HEAD
=======
#            # send to Expert?
#            if random.random() <= param.TRIA_REFERRAL[sample.referralType][sample.triageType]:
#                # send to expert.
#                logging.debug(' %4.1f %s  \tTO FIU.\t\t%s\tAverage time from start to T1: %3.2f min'
#                            %(self.env.now, self.name, sample.name, self.appTimeResolved/self.samples_resolved))
#                self.env.process(self.sent2FIU(sample, experts))
#
#            else:
#                # inspection is done.
#                self.samples_resolved += 1
#                appTime = self.env.now - sample.start
#                self.appTimeResolved += appTime
#                resolvedTimeList.append(appTime)
#                if peak_resolved_time < appTime:
#                    peak_resolved_time = appTime
#                logging.debug(' %4.1f %s  \tSOLVED.\t\t%s\tAverage time from start to Triage: %3.2f min'
#                            %(self.env.now, self.name, sample.name, self.appTimeResolved/self.samples_resolved))
#            # release the staff resource

>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
            logging.debug(' %4.1f %s  \tRELEASE\t\t%s' %(self.env.now, self.name, sample.name))

    def exitTriageTransfer(self, sample, experts):
        global peak_resolved_time

        if random.random() <= param.TRIA_REFERRAL[sample.referralType][sample.triageType][sample.matchType]:
            sample.potFraud = True
            while not self.open:
                # wait until next shift opens
                yield self.env.timeout(self.nextShift - self.env.now)

            if param.FIU_TRANSFER[sample.referralType][sample.triageType][sample.matchType] == 1:
                    self.samplesSendToFIU += 1
<<<<<<< HEAD
                    send2FIU(sample, experts, self.env.now, 'TRIAGE')
                    logging.debug(' %4.1f %s  \tTO TRIAGE.\t\t%sn' % (self.env.now, sample.name, sample.name))
                    self.env.process(experts.callFIU(sample))
            else:
                # blocked
                blockedDictTriage[sample.referralType] += 1
        else:
            # passport production
            passportDictTriage[sample.referralType] += 1



def send2FIU(sample, experts, now, name):
=======
                    sent2FIU(sample, experts, self.env.now, 'TRIAGE')
                    logging.debug(' %4.1f %s  \tTO TRIAGE.\t\t%sn' % (self.env.now, sample.name, sample.name))

                    self.env.process(experts.operation(sample))



def sent2FIU(sample, experts, now, name):
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
    global inQueueT2

    # increment No Referral count
    incDataframe(df_referralRate, 'Referral to FIU', sample.referralType)
    incDataframe(df_toFIU, sample.referralType, sample.triageType)
    incDataframe(df_impFIU, sample.referralType, sample.triageType)

    logging.debug(' %4.1f %s  \tSEND T2\t\t%s\tNumber in queue 2: %i'
                %(now, name, sample.name, inQueueT2))
    inQueueT2 += 1
    sample.type = 'T2'
    sample.enterT2 = now
    queueT2List.append((now, inQueueT2))


#############################################################################

class Expert(Staff):
    def __init__(self, env):
        Staff.__init__(self, env, param.SHIFT_EXPERT, param.SHIFT_EXPERT_ST, param.EXPERT_DAY_OFF, param.NUM_EXPERTS)
        self.name = 'EXPERT  '
        self.samples_examined = 0               # samples inspected from T1 queue
        self.samples_resolved

<<<<<<< HEAD
    def callFIU(self, sample):
=======
    def operation(self, sample):
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
        global inQueueT2, peak_resolved_time

        with self.staffRes.request(priority=1) as req:
            yield req
            # remove from queue T2
            logging.debug(' %4.1f %s  \tGET FROM T2\t%s\tinQueueT2: %i\ttype:%s'
                %(self.env.now, self.name, sample.name, inQueueT2, sample.type))
            inQueueT2 -= 1
            self.timeInQueueT2 += self.env.now - sample.enterT2
            self.acqQueueT2 += 1
            queueT2List.append((self.env.now, inQueueT2))
            if sample.impostor:
                impostorDictFIU[sample.referralType] += 1

<<<<<<< HEAD
            examination_time = numpy.abs(distributions('fiu_processing_time'))
=======
            examination_time = numpy.abs(distributions('fmt_processing_time'))
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
            yield self.env.timeout(examination_time)
            opTimeTotal[sample.referralType] += examination_time

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
<<<<<<< HEAD
        exitFIU(sample, self.env.now)
              
def exitFIU(sample, now):
    global peak_resolved_time

    if param.FIU_REFERRAL[sample.referralType][sample.triageType][sample.matchType] == 2 and sample.impostor:
        # blocked
        blockedDictFIU[sample.referralType] += 1
    else:
        # passport production
        passportDictFIU[sample.referralType] += 1
            
=======

>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
#############################################################################

class Sample(object):
    def __init__(self, name, referral=None, triage=None, match=None, impostor=False, sample_type='T1', start=0.0, enterT1=None, enterTriage=None, enterT2=None):
        self.name = name
        self.type = sample_type
        self.start = start
        self.referralType = referral
        self.triageType = triage
        self.matchType = match
        self.enterT1 = enterT1
        self.enterTriage = enterTriage
        self.enterT2 = enterT2
        self.impostor = impostor
        self.match = False
        self.potFraud = False

class Station(object):
    def __init__(self, env, name, examiners, triage, experts):
        self.env = env
        self.name = name.upper()
        self.bioEngine = simpy.Resource(env, capacity=1)
        #self.examiners = examiners
        #self.experts = experts

        self.referralType = ''                  # referral type AR, CR, FTA, CF, OTR
        self.triageType = ''                    # 1to1, 1toWL or 1toM
        self.matchType = ''                      # lights_out_match, possible_match, no_match
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
        self.open = False                       # if shift is open or closed

        env.process(self.shift(arrivalParam.OPEN_START*60, arrivalParam.OPEN_HOURS*60, arrivalParam.DAY_OFF, examiners, triage, experts))             # shift process


    def shift(self, shiftStart, shiftLenght, daysOff, examiners, triage, experts):
        # shift control
        yield self.env.timeout(shiftStart)

        while True:
            for day in arrivalParam.DAYS:
                if day not in daysOff:
                    # working day
                    self.open = True
                    self.env.process(self.createSample(examiners, triage, experts))
                    logging.debug(' %4.1f %s \tARRIVAL ON\t%s' %(self.env.now, self.name, day))
                    yield self.env.timeout(shiftLenght)
                    self.open = False
                    nextShift = self.env.now + 24*60 - shiftLenght
                    logging.debug(' %4.1f %s \tARRIVAL OFF' %(self.env.now, self.name))
                    yield self.env.timeout(max(nextShift - self.env.now,0))

                else:
                    # weekend
                    self.open = False
                    logging.debug(' %4.1f %s\t\tARRIVAL OFF WEEKEND\t%s' %(self.env.now, self.name, day))
                    nextShift = self.env.now + 24*60
                    yield self.env.timeout(max(nextShift - self.env.now,0))

    def createSample(self, examiners, triage, experts):
        """Create samples process"""
        while self.open:
            acquisition_time = numpy.abs(distributions('time_per_acquisition'))
            yield self.env.timeout(acquisition_time)
            self.samples_created += 1
            self.referralType = distributions('referral_type')
            populationTotal[self.referralType] += 1

            self.env.process(self.station_call('Sample %i' % self.samples_created, self.samples_created, examiners, triage, experts))

    def station_call(self, sample, numSample, examiners, triage, experts):
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

            # choose 1to1, 1toWL or 1toM
            self.triageType = choicesDist(param.REFERRAL_DEST, param.TRIAGE_PERC[self.referralType])

            if random.random() <= impostorProbDict[self.referralType]:
                # we have a impostor!
                self.impostor = True
                impostorDict[self.referralType] += 1

            #sample_score = distributions('score_per_sample')
            logging.debug(' %4.1f %s\tEND_MATCH\t%s\t\tTriage: %s' %(self.env.now, self.name, self.referralType, self.triageType))

        self.timeInStation += self.env.now - start

        if not self.triageType == 'free' and self.referralType in param.REFERRAL_SIM:
            self.samples_queued += 1

            # choose lights on, possible match or no_match
            self.matchType = choicesDist(param.PROCEDURE_TYPES, param.MATCHING[self.referralType][self.triageType].values())
<<<<<<< HEAD
            sample = Sample(sample, self.referralType, self.triageType, self.matchType, impostor=self.impostor, start=start, enterT1=self.env.now)
=======

>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
            if param.FMT_TRANSFER[self.referralType][self.triageType][self.matchType] == 1:
                if param.FMT_REFERRAL[self.referralType][self.triageType][self.matchType] == 1:
                    if param.TRIA_TRANSFER[self.referralType][self.triageType][self.matchType] == 1:
                        if param.TRIA_REFERRAL[self.referralType][self.triageType][self.matchType] == 1:
                            if param.FIU_TRANSFER[self.referralType][self.triageType][self.matchType] == 1:
<<<<<<< HEAD
                                send2FIU(sample, experts, self.env.now, self.name)
                                self.env.process(experts.callFIU(sample))

                        elif param.TRIA_REFERRAL[self.referralType][self.triageType][self.matchType] > 0:
                            send2Triage(sample, experts, self.env.now, self.name)
                            self.env.process(triage.callTriage(sample, experts))

                elif param.FMT_REFERRAL[self.referralType][self.triageType][self.matchType] > 0:
                    self.env.process(examiners.callFMT(sample, triage, experts))

            # blocked
            blockedDict[sample.referralType] += 1
=======
                                self.env.process(examiners.callFMT(Sample(sample, self.referralType, self.triageType, self.matchType, impostor=self.impostor, start=start, enterT1=self.env.now), triage, experts))

                        elif param.TRIA_REFERRAL[self.referralType][self.triageType][self.matchType] > 0:
                            self.env.process(examiners.callFMT(Sample(sample, self.referralType, self.triageType, self.matchType, impostor=self.impostor, start=start, enterT1=self.env.now), triage, experts))

                elif param.FMT_REFERRAL[self.referralType][self.triageType][self.matchType] > 0:
                    self.env.process(examiners.callFMT(Sample(sample, self.referralType, self.triageType, self.matchType, impostor=self.impostor, start=start, enterT1=self.env.now), triage, experts))

            # blocked

>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd



            # increment No Referral count
            incDataframe(df_referralRate, 'Referral to FMT', self.referralType)
            incDataframe(df_toFMT, self.referralType, self.triageType)
            incDataframe(df_impFMT, self.referralType, self.triageType)

            inQueueT1 += 1
            queueT1List.append((self.env.now, inQueueT1))


            # send to FMT
            #self.env.process(examiners.callFMT(Sample(sample, self.referralType, self.triageType, impostor=self.impostor, start=start, enterT1=self.env.now), triage, experts))

            logging.debug(' %4.1f %s\tFURTHER_I\t%s\tAverage time: %3.2f min'
                        %(self.env.now, self.name, sample, self.timeInStation/self.samples_acquired))
            self.instantStats()

        else:
<<<<<<< HEAD
            # passport production
            passportDict[self.referralType] += 1
=======
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
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
def createMulDict(typesDict, dataDict):
    d = {}
    for key, values in dataDict.items():
        dTmp = {}
        for key1, values1 in values.items():
            dTmp[key1] = dict(zip(typesDict, dataDict[key][key1]))

        d[key] = dTmp

    return d

def getParameters():

    ############################################################################
    print('\nAssigning the parameters ...')

    global impostorProbDict

   # get all simulation parameters from json file
    logging.debug('Biometric Examination Office \nCost-Staff Statistical Analysis\n')
    logging.debug('Reading of settings ...')

    # get input parameters from jason file
    import_settings=read_settings("Parameters_triage.json")
    # staffing
    full_time_equivalent_hours = int(import_settings["staffing_checking"]["full_time_equivalent_hours"])
<<<<<<< HEAD
    fmt_time_seconds = int(import_settings["staffing_checking"]["fmt_time_seconds"])
    triage_time_seconds = int(import_settings["staffing_checking"]["triage_time_seconds"])
    fiu_time_seconds = int(import_settings["staffing_checking"]["fiu_time_seconds"])
=======
    processing_time_seconds = int(import_settings["staffing_checking"]["processing_time_seconds"])
    triage_time_seconds = int(import_settings["staffing_checking"]["triage_time_seconds"])
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
    average_staff_absence_percentage = int(import_settings["staffing_checking"]["average_staff_absence_percentage"])
    impostor_percentage = import_settings["staffing_checking"]["impostor_percentage"]
    referral_percentage = import_settings["staffing_checking"]["referral_percentage"]

    param.TRIAGE_PERC = import_settings["staffing_checking"]["triage_percentage"]
    matching = import_settings["staffing_checking"]["matching"]
    FMT_transfer = import_settings["staffing_checking"]["FMT_transfer"]
    FMT_referral = import_settings["staffing_checking"]["FMT_referral"]
    triage_transfer = import_settings["staffing_checking"]["triage_transfer"]
    triage_referral = import_settings["staffing_checking"]["triage_referral"]
    FIU_transfer = import_settings["staffing_checking"]["FIU_transfer"]
    FIU_referral = import_settings["staffing_checking"]["FIU_referral"]

    # estimate for the 1-1toWL-1to1-1toM probability
    for key, value in param.TRIAGE_PERC.items():
        value = value + [1-sum(value)]
        param.TRIAGE_PERC[key] = value

    # watch_list
    imposter_rate_percent = float(import_settings["watch_list"]["imposter_rate_percent"])
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
    triage_day_off = int(import_settings["simulation_settings"]["triage_day_off"])

    shif_examiner_start = int(import_settings["simulation_settings"]["shif_examiner_start"])
    shif_expert_start = int(import_settings["simulation_settings"]["shif_expert_start"])
    shif_triage_start = int(import_settings["simulation_settings"]["shif_triage_start"])

    mean_acquisition_rate = float(import_settings["simulation_settings"]["mean_acquisition_rate"])
    examiner_cost = float(import_settings["simulation_settings"]["examiner_cost"])
    examiners_number = int(import_settings["simulation_settings"]["examiners_number"])
    examiners_shift = float(import_settings["simulation_settings"]["examiners_shift"])
    expert_cost = float(import_settings["simulation_settings"]["expert_cost"])
    experts_number = int(import_settings["simulation_settings"]["experts_number"])
    experts_shift = float(import_settings["simulation_settings"]["experts_shift"])
    triage_shift = float(import_settings["simulation_settings"]["triage_shift"])

    mean_escalation_rate = float(import_settings["simulation_settings"]["mean_scalation_rate"])
    simulation_time = float(import_settings["simulation_settings"]["simulation_time"])
    number_of_replications = int(import_settings["simulation_settings"]["number_of_replications"])
    arrivals_day_off = int(import_settings["simulation_settings"]["arrivals_day_off"])
    arrivals_shift = int(import_settings["simulation_settings"]["arrivals_shift"])
    arrivals_shif_start = int(import_settings["simulation_settings"]["arrivals_shif_start"])

    param.RANDOM_SEED = random_seed

    param.MATCH_MEAN = biometric_system_match_time/30                     # Biometric system match time
    param.NUM_EXAMINERS = examiners_number                                # Number of examiners in the office
    param.SHIFT_EXAMINER = examiners_shift  * 60                          # Length of examiners' shift in minutes
    param.COST_EXAMINER = examiner_cost/60                                # cost of an examiner per hour
    param.NUM_EXPERTS = experts_number                                    # Number of experts in the office
    param.SHIFT_EXPERT = experts_shift * 60                               # Length of experts' shift in minutes
    param.SHIFT_TRIAGE = triage_shift * 60                                # Length of triage' shift in minutes
    param.COST_EXPERT = expert_cost/60                                    # cost of an expert per hour
    param.SIM_TIME = simulation_time * 24 * 60                            # Simulation time in minutes
    param.NUM_DAYS = simulation_time
    param.ACQ_MEAN = 1.0 / (24*60/mean_acquisition_rate)                  # Mean acquisition rates (samples per day) for exponential distribution
    param.CALL_MEAN = mean_escalation_rate * param.ACQ_MEAN               # Mean time to call in minutes for exponential distribution
    param.SHIFT_EXAMINER_ST = shif_examiner_start*60                      # hour of shift start for examiners
    param.SHIFT_EXPERT_ST = shif_expert_start*60                          # hour of shift start for experts
    param.SHIFT_TRIAGE_ST = shif_triage_start*60                          # hour of shift start for triage

    param.MEAN_ESCALATION_RATE = mean_escalation_rate
    param.NUM_REP = number_of_replications                                # Number of replication
<<<<<<< HEAD
    
    param.FMT_MEAN_TIME = fmt_time_seconds/60.0                    # FMT processing time in minutes
    param.TRIAGE_MEAN_TIME = triage_time_seconds/60.0                     # FMT processing time in minutes
    param.FIU_MEAN_TIME = fiu_time_seconds/60.0                     # FMT processing time in minutes
 
    param.FMT_TIME = (param.FMT_MEAN_TIME, 0.1*param.FMT_MEAN_TIME)       # FMT distribution time parameters
    param.TRIAGE_TIME = (param.TRIAGE_MEAN_TIME, 0.1*param.TRIAGE_MEAN_TIME)       # FMT distribution time parameters
    param.FIU_TIME = (param.FIU_MEAN_TIME, 0.1*param.FIU_MEAN_TIME)       # FMT distribution time parameters
=======
    param.FMT_MEAN_TIME = processing_time_seconds/60.0                    # FMT processing time in minutes
    param.TRIAGE_MEAN_TIME = triage_time_seconds/60.0                     # FMT processing time in minutes
    param.FMT_TIME = (param.FMT_MEAN_TIME, 0.1*param.FMT_MEAN_TIME)       # FMT distribution time parameters
    param.TRIAGE_TIME = (param.TRIAGE_MEAN_TIME, 0.1*param.TRIAGE_MEAN_TIME)       # FMT distribution time parameters
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd

    param.REFERRAL_PROB = referral_percentage
    param.EXPERT_DAY_OFF = [i for i in range((len(param.DAYS) - experts_day_off + 1), len(param.DAYS) +1)]
    param.EXAMINER_DAY_OFF = [i for i in range((len(param.DAYS) - examiners_day_off + 1), len(param.DAYS) +1)]
    param.TRIAGE_DAY_OFF = [i for i in range((len(param.DAYS) - triage_day_off + 1), len(param.DAYS) +1)]

    param.MATCHING = createMulDict(param.PROCEDURE_TYPES, matching)
    param.FMT_TRANSFER = createMulDict(param.PROCEDURE_TYPES, FMT_transfer)
    param.FMT_REFERRAL = createMulDict(param.PROCEDURE_TYPES, FMT_referral)
    param.TRIA_TRANSFER = createMulDict(param.PROCEDURE_TYPES, triage_transfer)
    param.TRIA_REFERRAL = createMulDict(param.PROCEDURE_TYPES, triage_referral)
    param.FIU_TRANSFER = createMulDict(param.PROCEDURE_TYPES, FIU_transfer)
    param.FIU_REFERRAL = createMulDict(param.PROCEDURE_TYPES, FIU_referral)

    param.IMPOSTOR_PERC = impostor_percentage
    impostorProbDict = dict(zip(param.REFERRAL_TYPES,param.IMPOSTOR_PERC))

    arrivalParam.DAY_OFF = [i for i in range((len(arrivalParam.DAYS) - arrivals_day_off + 1), len(arrivalParam.DAYS) +1)]
    arrivalParam.OPEN_HOURS = arrivals_shift
    arrivalParam.OPEN_START = arrivals_shif_start
# Setup and start the simulation

def simulate():

    global inQueueT1, inQueueTriage, inQueueT2, queueT1List, queueTriageList, queueT2List
    global backlogT1List, backlogTriageList, backlogT2List, peak_resolved_time, resolvedTimeList

    peak_resolved_time = 0.0                                        # peak time taken for a sample to resolved
    resolvedTimeList = []                                           # list for time from acq to resolved
    inQueueT1, inQueueTriage, inQueueT2 = 0, 0, 0                                     # number in queue for Tiers
    queueT1List, queueTriageList, queueT2List = [(0,0)], [(0,0)], [(0,0)]                     # global list to store number in queue over time
    backlogT1List, backlogTriageList, backlogT2List = [], [] , []                          # global list to store backlog @shift begin



    #############################################################################
    #print('\nProcessing (%s-day period) ...' % simulation_time)
    logging.info(' Input parameters ...')
    logging.info(' Number of acquisition stations: %i'% param.NUM_STATIONS)
    logging.info(' Average of processing time in minutes: %3.2f', param.PT_MEAN)
    logging.info(' Deviation of processing time: %3.2f', param.PT_SIGMA)

    logging.info(' Number of examiners in the office: %i', param.NUM_EXAMINERS)
    logging.info(' Length of examiners shift in minutes: %3.2f', param.SHIFT_EXAMINER)
    logging.info(' Cost of an examiner per hour: %3.2f $/h', param.COST_EXAMINER)
    logging.info(' Number of experts in the office: %3.2f', param.NUM_EXPERTS)
    logging.info(' Length of experts shift in minutes: %3.2f', param.SHIFT_EXPERT)
    logging.info(' Cost of an expert per hour: %3.2f', param.COST_EXPERT)
    logging.info(' Simulation time in minutes: %3.2f', param.SIM_TIME)
    logging.info(' Minimum threshold: %3.2f', param.SCORE_MIN)
    logging.info(' Maximum threshold: %3.2f', param.SCORE_MAX)
    logging.info(' Mean acquisition rates for exponential distribution: %3.2f samples/min' %(param.ACQ_MEAN))

    logging.info(' Daily enrolment volume with facial biometric: %3.2f samples/day', param.ACQ_MEAN/(24*60))
    logging.info(' Biometric system match time: %3.2f seg', param.MATCH_MEAN*30)

    # Create an environment
    env = simpy.Environment()

    # Start the setup process
    experts = Expert(env)
    triage = Triage(env)
    examiners = Examiner(env)
    stations = [Station(env, 'Station %02d' % i, examiners, triage, experts) for i in range(1, param.NUM_STATIONS+1)]

    # Execute!
    logging.info(' %4.1f Processing (%s-day period) ...' %(env.now, param.SIM_TIME/(24*60)))

    env.run(until=param.SIM_TIME)

    #print('\nOverall statistics ...\n')
    logging.info('\nOverall statistics ...')
    examiners_samples = examiners.samples_inspected
    # appTimeResolved: time from sample create to be resolved (leave the system by T1 and/or T2)
    examiners_times = examiners.appTimeResolved
    examiners_cost = examiners.busyTime*param.COST_EXAMINER
    #examiners_rate = examiners.busyTime/examiners_samples if examiners_samples > 0 else 0.0
    examiners_busy = examiners.busyTime/param.NUM_EXAMINERS
    examiners_utilization = examiners_busy/examiners.time2op

    experts_samples = experts.samples_inspected + experts.samples_examined
    experts_inspected = experts.samples_inspected

    experts_times = experts.appTimeResolved
    experts_cost = experts.busyTime*param.COST_EXPERT
    #experts_rate = experts.busyTime/experts_samples if experts_samples > 0 else 0.0
    experts_busy = experts.busyTime/param.NUM_EXPERTS
    experts_utilization = experts_busy/experts.time2op

    # all samples examined from T1: examiners and experts
    examined_samples = examiners_samples + experts.samples_examined
    triage_inspected = triage.samples_inspected

    created_samples = numpy.sum(station.samples_created for station in stations)
    acquired_samples = numpy.sum(station.samples_acquired for station in stations)
    waiting_samples = numpy.sum(len(station.bioEngine.queue) for station in stations)
    passed_samples = numpy.sum(station.samples_passed for station in stations)
    queued_samples = numpy.sum(station.samples_queued for station in stations)

    biometric_queue_time = numpy.sum(station.acqQueueTime for station in stations)/acquired_samples
    biometric_utilization = numpy.sum(station.bioEngBusyTime for station in stations)/param.SIM_TIME/param.NUM_STATIONS
    biometric_queue_avg = param.ACQ_MEAN * biometric_queue_time  # little´s law

    avg_resolved_time = (examiners.appTimeResolved + experts.appTimeResolved+ triage.appTimeResolved)/(examiners.samples_resolved + experts.samples_resolved+ triage.samples_resolved)

    total_times = numpy.sum([examiners_times,experts_times])
    total_costs = numpy.sum([examiners_cost,experts_cost])
    total_rates = (examiners.samples_resolved + experts.samples_resolved)/(param.SIM_TIME)
    #total_rates = numpy.sum([examiners_samples*examiners_rate,experts_samples*experts_rate])/numpy.sum([examiners_samples,experts_samples])

    queue_T1_avg_time = (examiners.timeInQueueT1 + experts.timeInQueueT1)/(examiners.acqQueueT1 + experts.acqQueueT1) if examiners.acqQueueT1 + experts.acqQueueT1>0 else 0.0
    queue_Triage_avg_time = (experts.timeInQueueTriage)/(experts.acqQueueTriage) if experts.acqQueueTriage > 0 else 0.0
    queue_T2_avg_time = (experts.timeInQueueT2)/(experts.acqQueueT2) if experts.acqQueueT2 > 0 else 0.0

    examiners_mean_backlog = 24 * 60 * (queued_samples - examined_samples)/param.SHIFT_EXAMINER
    #examiners_mean_backlog = 24 * 60 * (queued_samples - examiners_samples)/SHIFT_EXAMINER

    experts_mean_backlog = 24 * 60 * (examiners.samplesSendToTriage - experts_inspected)/param.SHIFT_EXPERT
<<<<<<< HEAD
    triage_mean_backlog = 24 * 60 * (triage.samplesSendToFIU - triage_inspected)/param.SHIFT_TRIAGE
=======
    triage_mean_backlog = 24 * 60 * (triage.samplesSendToExperts - triage_inspected)/param.SHIFT_TRIAGE
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd

    #examiners_mean_backlog = numpy.mean(backlogT1List)
    #experts_mean_backlog =  numpy.mean(backlogT2List)

    queueT1List.append((env.now, inQueueT1))
    queueTriageList.append((env.now, inQueueTriage))
    queueT2List.append((env.now, inQueueT2))

    examiners_mean_queue = queueAvg(queueT1List)
    triage_mean_queue = queueAvg(queueTriageList)
    experts_mean_queue = queueAvg(queueT2List)

    queue_length_at_end = queued_samples - (examiners.samples_resolved + experts.samples_resolved)

    output = []

    fields = ["Sample_Acquisition_Rate",
              "Expert_Inspection_Cost",
              "Experts_Mean_Backlog",
              "Examiner_Processing_Cost",
              "Examiners_Mean_Backlog",
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

    output.append({"Sample_Acquisition_Rate": param.ACQ_MEAN/(24*60),
                   "Expert_Inspection_Cost": param.COST_EXPERT,
                   "Experts_Mean_Backlog": experts_mean_backlog,
                   "Examiner_Processing_Cost": param.COST_EXAMINER,
                   "Examiners_Mean_Backlog": examiners_mean_backlog,
                   "Experts_Number": param.NUM_EXPERTS,
                   "Examiners_Number": param.NUM_EXAMINERS,
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

    logging.info(' Sample_Acquisition_Rate:  %4.1f samples/day' %( param.ACQ_MEAN/(24*60)))
    logging.info(' Acquisition_Rate:  %4.2f samples/day' %(queued_samples/ param.ACQ_MEAN/(24*60)))
    logging.info(' Expert_Inspection_Cost:  $%4.1f' %(param.COST_EXPERT))
    logging.info(' Experts_Mean_Backlog:  %4.1f samples' %(experts_mean_backlog))
    logging.info(' Examiner_Processing_Cost:  $%4.1f' %(param.COST_EXAMINER))
    logging.info(' Examiners_Mean_Backlog:  %4.1f samples' %(examiners_mean_backlog))
    logging.info(' Experts_Number:  %4.1f experts' %(param.NUM_EXPERTS))
    logging.info(' Examiners_Number:  %4.1f examiners' %(param.NUM_EXAMINERS))
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



    return fields, output, queueT1List, queueTriageList, queueT2List


if __name__ == '__main__':

    def sumReplication(data, column):
        # sum dataframe column
        df_Tmp = pd.DataFrame([data])
        df_Tmp = df_Tmp[param.REFERRAL_TYPES]
        df_Tmp = df_Tmp.transpose()
        df_popSim[column] = df_popSim[column] + df_Tmp[df_Tmp.columns[0]]/param.NUM_DAYS

    df_popSim = createDF(param.REFERRAL_TYPES,
                         ['Total Population', 'Total Impostors',
                          'Type Percentage', 'Impostors Percentage',
                          'Total FMT Time (min)', 'Total Triage Time (min)', 'Total FIU Time (min)',
<<<<<<< HEAD
                          'Passport production @Match', 'Passport production @FMT', 'Passport production @Triage', 'Passport production @FIU',
                          'Passport blocked @Match', 'Passport blocked @FMT', 'Passport blocked @Triage', 'Passport blocked @FIU',
                          'Total Impostors @FMT', 'Total Impostors @Triage', 'Total Impostors @FIU'],
=======
                          'Total Impostors @FIU'],
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
                          'int32')

    df_toFMTSim = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'float32')
    df_toTriageSim = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'float32')
    df_toFIUSim = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'float32')
    df_AtTimeSim = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'float32')
    df_TriageTimeSim = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'float32')
    df_OpTimeSim = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'float32')

    getParameters()
    random.seed(param.RANDOM_SEED)  # This helps reproducing the results

    plt.close('all')
    fig, axes = plt.subplots(ncols=1, nrows=3)
    ax1, ax2, ax3 = axes.ravel()

    for rep in range(param.NUM_REP):
        print('Replication', rep)
<<<<<<< HEAD
        passportDict = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        passportDictFMT = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        passportDictTriage = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        passportDictFIU = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))

        blockedDict = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))        
        blockedDictFMT = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        blockedDictTriage = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        blockedDictFIU = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        
=======
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
        impostorDict = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        impostorDictFMT = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        impostorDictTriage = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        impostorDictFIU = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
<<<<<<< HEAD
        
=======
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
        populationTotal = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        atTimeTotal = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        triageTimeTotal = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
        opTimeTotal = dict(zip(param.REFERRAL_TYPES, [0, 0, 0, 0, 0]))
<<<<<<< HEAD
        
=======
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd
        df_referralRate = createDF(['No Referral', 'Referral to FMT', 'Referral to FIU'], param.REFERRAL_TYPES, 'int32')
        df_toFMT = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'int32')
        df_toTriage = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'int32')
        df_toFIU = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'int32')
        df_impFMT = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'int32')
        df_impTriage = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'int32')
        df_impFIU = createDF(param.REFERRAL_TYPES, param.FMT_TYPES, 'int32')

        fields, output, queueT1List, queueTriageList, queueT2List = simulate()
        queuePlot(queueT1List, ax1, (param.SHIFT_EXAMINER_ST, (param.SHIFT_EXAMINER_ST + param.SHIFT_EXAMINER)), 'Replication ' + str(rep))
        queuePlot(queueTriageList, ax2, (param.SHIFT_TRIAGE_ST, (param.SHIFT_TRIAGE_ST + param.SHIFT_TRIAGE)), 'Replication ' + str(rep))
        queuePlot(queueT2List, ax3, (param.SHIFT_EXPERT_ST, (param.SHIFT_EXPERT_ST + param.SHIFT_EXPERT)), 'Replication ' + str(rep))

        sumReplication(populationTotal,'Total Population')
<<<<<<< HEAD
        
        sumReplication(passportDict, 'Passport production @Match')
        sumReplication(passportDictFMT,'Passport production @FMT')
        sumReplication(passportDictTriage,'Passport production @Triage')
        sumReplication(passportDictFIU,'Passport production @FIU')

        sumReplication(blockedDict, 'Passport blocked @Match')
        sumReplication(blockedDictFMT,'Passport blocked @FMT')
        sumReplication(blockedDictTriage,'Passport blocked @Triage')
        sumReplication(blockedDictFIU,'Passport blocked @FIU')
        
        sumReplication(impostorDict,'Total Impostors')
        sumReplication(impostorDictFMT,'Total Impostors @FMT')
        sumReplication(impostorDictTriage,'Total Impostors @Triage')
        sumReplication(impostorDictFIU,'Total Impostors @FIU')
        
        sumReplication(atTimeTotal,'Total FMT Time (min)')
        sumReplication(triageTimeTotal,'Total Triage Time (min)')
        sumReplication(opTimeTotal,'Total FIU Time (min)')
        
=======
        sumReplication(impostorDict,'Total Impostors')
        sumReplication(atTimeTotal,'Total FMT Time (min)')
        sumReplication(triageTimeTotal,'Total Triage Time (min)')
        sumReplication(opTimeTotal,'Total FIU Time (min)')
        sumReplication(impostorDictFIU,'Total Impostors @FIU')
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd

        df_toFMTSim = df_toFMTSim + df_toFMT/param.NUM_DAYS
        df_toTriageSim = df_toTriageSim + df_toFMT/param.NUM_DAYS
        df_toFIUSim = df_toFIUSim + df_toFIU/param.NUM_DAYS

    fig.suptitle("Biometric Examination Office, Cost-Staff Statistical Analysis")
    ax1.set_xlabel("Simulation time (days)")
    ax2.set_xlabel("Simulation time (days)")
    ax2.set_xlabel("Simulation time (days)")
    ax1.set_ylabel("Referrals in FMT queue")
    ax2.set_ylabel("Referrals in Triage queue")
    ax3.set_ylabel("Referrals in FIU queue")
    plt.show()
    df_popSim = df_popSim/param.NUM_REP
    df_toFMTSim = df_toFMTSim/param.NUM_REP
    df_toTriageSim = df_toTriageSim/param.NUM_REP
    df_toFIUSim = df_toFIUSim/param.NUM_REP

    columnsNames = [i+'_FMT' for i in param.FMT_TYPES]
    df_toFMTSim.columns = columnsNames

    columnsNames = [i+'_Triage' for i in param.FMT_TYPES]
    df_toTriageSim.columns = columnsNames

    columnsNames = [i+'_FIU' for i in param.FMT_TYPES]
    df_toFIUSim.columns = columnsNames

    totalTmp = numpy.sum(df_popSim['Total Population'])
    df_popSim['Type Percentage'] = df_popSim['Total Population']/totalTmp

    df_popSim['Impostors Percentage'] = df_popSim['Total Impostors']/df_popSim['Total Population']

    writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

    df_popSim.to_excel(writer)
<<<<<<< HEAD
    df_toFMTSim.to_excel(writer, startrow=10)
    df_toTriageSim.to_excel(writer, startrow=20)
    df_toFIUSim.to_excel(writer, startrow=30)
=======
    df_toFMTSim.to_excel(writer, startcol=10)
    df_toTriageSim.to_excel(writer, startcol=15)
    df_toFIUSim.to_excel(writer, startcol=21)
>>>>>>> bf78ff7e8f88bd07022aafa447e57498b88701bd

    writer.save()