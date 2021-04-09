import pm4py
import os

import itertools as it
import pandas as pd

from operator import itemgetter
from datetime import timedelta

from Utils import FileUtils, PrintUtils
import Data.LogSplitter as ls


class DataLoader(object):

    def __init__(self, filePath: str, timeformat: str, column_names: dict(str, str),  one_timeStamp: bool, filter_d_attrib: bool, verbose: bool = False, append_start_end: bool = False) -> None:
        self.file_path: str = filePath
        self.file_name, self.file_extension = FileUtils.define_file_type(
            self.file_path)
        self.column_names: dict(str, str) = column_names
        self.one_timeStamp: bool = one_timeStamp
        self.filter_d_attrib: bool = filter_d_attrib
        self.verbose: bool = verbose
        self.timeformat = timeformat
        self.append_start_end = append_start_end

        # Initialise data fields

        self.data = list()
        self.rawData = list()
        self.__loadDataFromFile()
        self.__preprocess()

    def __loadDataFromFile(self):
        """
        reads all the data from the log depending
        the extension of the file
        """
        if self.file_extension == '.xes':
            self.__get_xes_events_data()
        elif self.file_extension == '.csv':
            self.__get_csv_events_data()
        
        self.log = pd.DataFrame(self.data)

    def __get_xes_events_data(self):
        log = pm4py.read_xes(self.file_name)
        try:
            source = log.attributes['source']
        except:
            source = ''
        flattern_log = ([{**event,
                          'caseid': trace.attributes['concept:name']}
                         for trace in log for event in trace])
        temp_data = pd.DataFrame(flattern_log)
        temp_data['time:timestamp'] = temp_data.apply(
            lambda x: x['time:timestamp'].strftime(self.timeformat), axis=1)
        temp_data['time:timestamp'] = pd.to_datetime(temp_data['time:timestamp'],
                                                     format=self.timeformat)
        temp_data.rename(columns={
            'concept:name': 'task',
            'lifecycle:transition': 'event_type',
            'org:resource': 'user',
            'time:timestamp': 'timestamp'}, inplace=True)
        temp_data = (temp_data[~temp_data.task.isin(
            ['Start', 'End', 'start', 'end'])].reset_index(drop=True))
        temp_data = (
            temp_data[temp_data.event_type.isin(['start', 'complete'])]
            .reset_index(drop=True))
        if source == 'com.qbpsimulator':
            if len(temp_data.iloc[0].elementId.split('_')) > 1:
                temp_data['etype'] = temp_data.apply(
                    lambda x: x.elementId.split('_')[0], axis=1)
                temp_data = (
                    temp_data[temp_data.etype == 'Task'].reset_index(drop=True))
        self.rawData = temp_data.to_dict('records')
        if self.verbose:
            PrintUtils.printPerformedTask('Rearranging log traces ')
        self.data = self.__reorder_xes(temp_data)
        if (self.append_start_end):
            self.__append_start_end_event()
        if self.verbose:
            PrintUtils.printDoneTask()

    def __reorder_xes(self, temp_data):
        """
        this method match the duplicated events on the .xes log
        """
        temp_data = pd.DataFrame(temp_data)
        ordered_event_log = list()
        if self.one_timeStamp:
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            temp_data = temp_data[temp_data.event_type == 'complete']
            ordered_event_log = temp_data.rename(
                columns={'timestamp': 'end_timestamp'})
            ordered_event_log = ordered_event_log.drop(columns='event_type')
            ordered_event_log = ordered_event_log.to_dict('records')
        else:
            self.column_names['Start Timestamp'] = 'start_timestamp'
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            for caseid, group in temp_data.groupby(by=['caseid']):
                trace = group.to_dict('records')
                temp_trace = list()
                for i in range(0, len(trace)-1):
                    incomplete = False
                    if trace[i]['event_type'] == 'start':
                        c_task_name = trace[i]['task']
                        remaining = trace[i+1:]
                        complete_event = next((event for event in remaining if (
                            event['task'] == c_task_name and event['event_type'] == 'complete')), None)
                        if complete_event:
                            temp_trace.append(
                                {'caseid': caseid,
                                 'task': trace[i]['task'],
                                 'user': trace[i]['user'],
                                 'start_timestamp': trace[i]['timestamp'],
                                 'end_timestamp': complete_event['timestamp']})
                        else:
                            incomplete = True
                            break
                if not incomplete:
                    ordered_event_log.extend(temp_trace)
        return ordered_event_log

    def __get_csv_events_data(self):
        """
        reads and parse all the events information from a csv file
        """
        if self.verbose:
            PrintUtils.printPerformedTask('Reading log traces ')
        log = pd.read_csv(self.file_path)
        if self.one_timeStamp:
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user', 'end_timestamp']]
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        else:
            self.column_names['Start Timestamp'] = 'start_timestamp'
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user',
                           'start_timestamp', 'end_timestamp']]
            log['start_timestamp'] = pd.to_datetime(log['start_timestamp'],
                                                    format=self.timeformat)
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        self.data = log.to_dict('records')
        if self.append_start_end:
            self.__append_start_end_event()
        self.__split_event_transitions()
        if self.verbose:
            PrintUtils.printDoneTask()

    def __split_event_transitions(self):
        temp_raw = list()
        if self.one_timeStamp:
            for event in self.data:
                temp_event = event.copy()
                temp_event['timestamp'] = temp_event.pop('end_timestamp')
                temp_event['event_type'] = 'complete'
                temp_raw.append(temp_event)
        else:
            for event in self.data:
                start_event = event.copy()
                complete_event = event.copy()
                start_event.pop('end_timestamp')
                complete_event.pop('start_timestamp')
                start_event['timestamp'] = start_event.pop('start_timestamp')
                complete_event['timestamp'] = complete_event.pop(
                    'end_timestamp')
                start_event['event_type'] = 'start'
                complete_event['event_type'] = 'complete'
                temp_raw.append(start_event)
                temp_raw.append(complete_event)
        self.raw_data = temp_raw

    def __append_start_end_event(self):
        end_start_times = dict()
        for case, group in pd.DataFrame(self.data).groupby('caseid'):
            end_start_times[(case, 'Start')] = (
                group.start_timestamp.min()-timedelta(microseconds=1))
            end_start_times[(case, 'End')] = (
                group.end_timestamp.max()+timedelta(microseconds=1))
        new_data = list()
        data = sorted(self.data, key=lambda x: x['caseid'])
        for key, group in it.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = end_start_times[(key, new_event)]
                if not self.one_timeStamp:
                    temp_event['start_timestamp'] = end_start_times[(
                        key, new_event)]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        self.data = new_data

    @staticmethod
    def create_index(log_df, column):
        """Creates an idx for a categorical attribute.
        parms:
            log_df: dataframe.
            column: column name.
        Returns:
            index of a categorical attribute pairs.
        """
        temp_list = log_df[[column]].values.tolist()
        subsec_set = {(x[0]) for x in temp_list}
        subsec_set = sorted(list(subsec_set))
        alias = dict()
        for i, _ in enumerate(subsec_set):
            alias[subsec_set[i]] = i + 1
        return alias

    def __indexing(self):
            # Activities index creation
        self.ac_index = self.create_index(self.log, 'task')
        self.ac_index['start'] = 0
        self.ac_index['end'] = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}
        # Roles index creation
        self.rl_index = self.create_index(self.log, 'role')
        self.rl_index['start'] = 0
        self.rl_index['end'] = len(self.rl_index)
        self.index_rl = {v: k for k, v in self.rl_index.items()}
        # Add index to the event log
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log['ac_index'] = self.log.apply(ac_idx, axis=1)
        rl_idx = lambda x: self.rl_index[x['role']]
        self.log['rl_index'] = self.log.apply(rl_idx, axis=1)

    def __preprocess(self ):
        # indexes creation
        self.__indexing()
        # split validation
        self.__split_timeline(0.8, self.one_timeStamp)

    def __split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """

        # Split log data
        splitter = ls.LogSplitter(self.log)
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events*0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = (train.sort_values(key, ascending=True)
                          .reset_index(drop=True))



# =============================================================================
# Accesssor methods
# =============================================================================
    def get_traces(self):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in self.data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if self.one_timeStamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), self.data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

    def get_raw_traces(self):
        """
        returns the raw data splitted by caseid and ordered by timestamp
        """
        cases = list(set([c['caseid'] for c in self.raw_data]))
        traces = list()
        for case in cases:
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), self.raw_data)),
                key=itemgetter('timestamp'))
            traces.append(trace)
        return traces

    def set_data(self, data):
        """
        seting method for the data attribute
        """
        self.data = data
