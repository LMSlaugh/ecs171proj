# class under development

'''
This class uses access the PI web API through Python

It replicates the functions of PI datalink excel Add-on (Windows only) in Python

Some features may be UC Davis specific, but can easily be extended to other PI installations

v.0.7 (Nov 26 2017)
- fixed broken "interval" arg
- added fixed maxCount = 100,000 for recorded calculation
- removed all _ in the name of the arguments
- changed get_stream name to _get_stream since it's almost never used out from the outside
- added option "calculated" to calculation arg (similar to PI datalink), now default for get_stream_by_point and by_path
- added def parse_summary_duration for this option
- changed get_stream to add calculated option
- changed compose_stream_url to add calculated option
- added "selectedFields" parameter to minimize data returned in the json

v.0.6
-fixed timezone bug

v0.5
-reorganized code: help methods first
-added path search - this actually requires four different methods to parse the object returned because they are different than the serach by point
(get_stream_by_path, get_webID_by_path, search_by_path, _parse_path)

v0.4
-calculation default tosumType=All
-removed the printing statement of point name and webID

v0.3
-returns DataFrame for all get options
-added a method to parse names

v0.1 
-renamed PI_downloader to PIPy_Datalink (similar to the excel plugin)

BUGS:
-when using labels arg, path should return the column with label name
-lables for multi-series don't work

TODO:
-handle "Index out of range " error when sending a query by name with "+" to the API that returns error


@author Marco Pritoni <marco.pritoni@gmail.com>
@author + EEC Project 3 team

latest update: Nov 26 2017 

'''

import pandas as pd
import os
import requests as req
import json
import numpy as np
import pytz


class pi_client(object):

    def __init__(self, root=None, calculation=None, interval=None, buildingMeterDB=None):

        if root == None:

            # for more general application this root should be read from config file in not set as an arg
            self.root = "https://ucd-pi-iis.ou.ad3.ucdavis.edu/piwebapi/"


    """

    The following section contains the parsers for the response of the PI server

    """


    def _parse_point(self, response, include_Path):
        """
        Example API json returned:

        {
          "Items": [
            {
              "WebId": "P09KoOKByvc0-uxyvoTV1UfQ61oCAAVVRJTC1QSS1QXFBFUy5BSFUuQ09PTElORyBFTkVSR1kgQlRVIFBFUiBIUg",
              "Name": "PES.AHU.Cooling Energy BTU per Hr",
              'Path': u'\\\\UTIL-PI-P\\PES.AHU.Cooling Energy BTU per Hr',
            },
            {
              "WebId": "P09KoOKByvc0-uxyvoTV1UfQ7FoCAAVVRJTC1QSS1QXFBFUy5BSFUuSEVBVElORyBFTkVSR1kgQlRVIFBFUiBIUg",
              "Name": "PES.AHU.Heating Energy BTU per Hr",
              'Path': u'\\\\UTIL-PI-P\\PES.AHU.Heating Energy BTU per Hr'
            }
        }
        """

        js = json.loads(response.text)

        # save results in a list of points and a dict of {point_name: Web_ID}
        point_list = []
        point_dic = {}

        # PARSE what is returned
        # for each element in the json structure
        for elem in range(0, len(js["Items"])):

            Point_name_full = js["Items"][elem]["Name"]  # see example
            point_list.append(Point_name_full)

            # also returns dict
            curr_point_dic = {}

            curr_point_dic["WebId"] = js["Items"][elem]["WebId"]


            if include_Path:
                curr_point_dic["Path"] = js["Items"][elem]["Path"]

            point_dic[Point_name_full] = curr_point_dic

        try:
            return point_list, point_dic
        except:
            return [], {} 

    def _parse_path(self, response, include_Path):
        """
        Example API json returned:


        {
          "WebId": "A0EbgZy4oKQ9kiBiZJTW7eugwS5GAMtE55BGIPhgDcyrprwcrOde7rrSVobodgP17EChQVVRJTC1BRlxDRUZTXFVDREFWSVNcQlVJTERJTkdTXFJJRkxFIFJBTkdFXEVMRUNUUklDSVRZfERFTUFORF9LQlRV",
          "Name": "Demand_kBtu"
        }

        NOTE: this is different from the JSON obtained by search_by_point
        Only single WebId and Name are returned in response. Parsing accordingly.
        Kept variable names same as _parse_point. Did not need to update since only used in local context.
        """

        js = json.loads(response.text)

        # save results in a list of points and a dict of {point_name: Web_ID}
        point_list = []
        point_dic = {}

        # PARSE what is returned

        # Have yet to encounter multiple return Items so removed for_loop indexing
        Point_name_full = js["Name"]
        point_list.append(Point_name_full)

        # can also return dict
        curr_point_dic = {}
        curr_point_dic["WebId"] = js["WebId"]

        if include_Path:
            curr_point_dic["Path"] = js["Path"]

        point_dic[Point_name_full] = curr_point_dic


        try:
            return point_list, point_dic
        except:
            return [], {}

    def _parse_TS(self, response, Web_ID, label):
        """
        Example API json parsed:

        {
          "Links": {},
          "Items": [
            {
              "Timestamp": "2017-02-10T02:45:00.2475263Z",
              "Value": 75.20761,
              "UnitsAbbreviation": "",
              "Good": true,
              "Questionable": false,
              "Substituted": false
            },
            {
              "Timestamp": "2017-02-10T03:45:00.2475263Z",
              "Value": 75.19933,
              "UnitsAbbreviation": "",
              "Good": true,
              "Questionable": false,
              "Substituted": false
            },
        ...
        Note that a subset of fields are actually returned thanks to the parameter: 
        "selectedFields": "Items.Timestamp;Items.Value"
        
        
        }
        """
        if response:
            # loads content of json response
            js = json.loads(response.text)

            timeseries = {}
            # counts the elements
            n_elem = len(js["Items"])

            # loops through the json in search of timestep and value - note this is not vectorized as the structure is irregular
            for i in range(0, n_elem):

                # saves timestep
                timestamp = js["Items"][i]["Timestamp"]

                # saves value - unless the calculated value is missing (failed calculation)
                value = js["Items"][i]["Value"]
                try:
                    # format to float
                    float(value)
                    pass
                except:
                    # if calculation failed can ignore the results (nan)
                    value = np.nan #value #np.nan
                    # or can get the default value: fixed
                    # value=js["Items"][i]["Value"]["Value"]

                # saves timeseries and value in a dictionary
                timeseries[timestamp] = value

            # converts dict into pandas series
            ts = pd.Series(timeseries)

            # converts Series index to datetime type
            ts.index = pd.to_datetime(ts.index)

            # saves name of the Series
            if label:

                # uses lable provided
                ts.name = label

            else:

                # uses the WebID if label not provided
                ts.name = Web_ID
            ts = pd.DataFrame(ts)

            # Adjusting timestamp
            ts = self._utc_to_local(ts, 'America/Los_Angeles')

            # Remove duplicates in the index
            ts = ts[~ts.index.duplicated(keep='first')]

        # If the requests fails
        else:
            # print to screen error
            print ("I can't find the stream with this WebID")

            # returns empty Dataframe
            return pd.DataFrame()

        return ts

    def _parse_summary(self, response, Web_ID, label):
        """
        Example API json parsed:

        {
          "Links": {},
          "Items": [
            {
              "Type": "Total",
              "Value": {
                "Timestamp": "2017-02-10T04:09:00.7909406Z",
                "Value": 75.166832186264742,
                "UnitsAbbreviation": "",
                "Good": true,
                "Questionable": false,
                "Substituted": false
              }
            },
            {
              "Type": "Average",
              "Value": {
                "Timestamp": "2017-02-10T04:09:00.7909406Z",
                "Value": 75.166832186264742,
                "UnitsAbbreviation": "",
                "Good": true,
                "Questionable": false,
                "Substituted": false
              }
            },
         ...
        """
        if response:
            # loads content of json response
            js = json.loads(response.text)

            # counts the elements
            n_elem = len(js["Items"])
            summary_dic = {}

            # loops through the json to extract each summary value
            for i in range(0, n_elem):
                sumType = js["Items"][i]['Type']
                SumVal = js["Items"][i]['Value']['Value']
                summary_dic[sumType] = SumVal

            df_summ = pd.DataFrame.from_dict(summary_dic, orient='index')

            if label:
                df_summ.columns = [label]

            else:
                df_summ.columns = [Web_ID]

            return df_summ

        else:
            # print error to screen
            print ("I can't find the stream with this WebID")
        return pd.DataFrame()

    def _parse_summary_with_duration(self, response, Web_ID, label):
        """
        Example API json parsed:

{
    "Links": {},
    "Items": [
        {
            "Type": "Average",
            "Value": {
                "Timestamp": "2017-11-20T08:00:00Z",
                "Value": 992.66693751017249,
                "UnitsAbbreviation": "",
                "Good": true,
                "Questionable": false,
                "Substituted": false
            }
        },
        {
            "Type": "Average",
            "Value": {
                "Timestamp": "2017-11-20T09:00:00Z",
                "Value": 979.03958468967016,
                "UnitsAbbreviation": "",
                "Good": true,
                "Questionable": false,
                "Substituted": false
            }
        },
        {
            "Type": "Average",
            "Value": {
                "Timestamp": "2017-11-20T10:00:00Z",
                "Value": 950.58930477566184,
                "UnitsAbbreviation": "",
                "Good": true,
                "Questionable": false,
                "Substituted": false
            }
        },
        ...
        Note that a subset of fields are actually returned thanks to the parameter: 
        "selectedFields": "Items.Type;Items.Value.Timestamp;Items.Value.Value"

        """
        if response:
            # loads content of json response
            js = json.loads(response.text)

            # counts the elements
            n_elem = len(js["Items"])
            summary_dic = {}
            timeseries = {}
            
            sumType = js["Items"][0]['Type'] # first value for sumType == "All"
            
            for i in range(0, n_elem):
                
                # we are using summary with duration to extract a time series
                # this supports only Average time series
            
                if js["Items"][i]['Type'] == sumType :
                    
                    # saves timestep
                    timestamp = js["Items"][i]['Value']["Timestamp"]

                    # saves value - unless the calculated value is missing (failed calculation)
                    value = js["Items"][i]["Value"]['Value']
                    try:
                        # format to float
                        float(value)
                        pass
                    except:
                        # if calculation failed can ignore the results (nan)
                        value = np.nan
                        # or can get the default value: fixed
                        # value=js["Items"][i]["Value"]["Value"]

                    # saves timeseries and value in a dictionary
                    timeseries[timestamp] = value

            ts = pd.Series(timeseries)

            # converts Series index to datetime type
            ts.index = pd.to_datetime(ts.index)

            # saves name of the Series
            if label:

                # uses lable provided
                ts.name = label

            else:

                # uses the WebID if label not provided
                ts.name = Web_ID
            ts = pd.DataFrame(ts)

            # Adjusting timestamp
            ts = self._utc_to_local(ts, 'America/Los_Angeles')

            # Remove duplicates in the index
            ts = ts[~ts.index.duplicated(keep='first')]

        # If the requests fails
        else:
            # print to screen error
            print ("I can't find the stream with this WebID")

            # returns empty Dataframe
            return pd.DataFrame()

        return ts 

    def _parse_end(self, response, Web_ID, label):
        """
        Example API json parsed:
        {
          "Timestamp": "2017-02-10T07:59:00Z",
          "Value": 75.1643,
          "UnitsAbbreviation": "",
          "Good": true,
          "Questionable": false,
          "Substituted": false
        }

        """
        if response:
            # loads content of json response
            js = json.loads(response.text)

            end_dic = {}

            # save json object in a dictionary
            end_dic['Good'] = js['Good']
            end_dic['Timestamp'] = js['Timestamp']
            end_dic['Value'] = js['Value']

            df_end = pd.DataFrame.from_dict(end_dic, orient='index')

            if label:
                df_end.columns = [label]

            else:
                df_end.columns = [Web_ID]

            return df_end

        else:
            # print to screen error
            print ("I can't find the stream with this WebID")

        return js['Value']

    """

    The following section contains methods to manage timezones

    """

    def _utc_to_local(self, data, local_zone):
        '''
        Function takes in pandas dataframe and adjusts index according to timezone in which is requested by user

        Parameters
        ----------
        data: Dataframe
            pandas dataframe of json timeseries response from server

        local_zone: string
            pytz.timezone string of specified local timezone to change index to

        Returns
        -------
        data: Dataframe
            Pandas dataframe with timestamp index adjusted for local timezone
        '''
        data.index = data.index.tz_localize(pytz.utc).tz_convert(
            local_zone)  # accounts for localtime shift
        # Gets rid of extra offset information so can compare with csv data
        data.index = data.index.tz_localize(None)

        return data

    # Change timestamp request time to reflect request in terms of local time relative to utc - working as of 5/5/17 ( Should test more )
    def _local_to_utc(self, timestamp, local_zone):
        """
        This method loads content of json response for a time series data of a single meter
        It also corrects for time zone of the response
        """

        # pacific = pytz.timezone('US/Pacific') # Setting timezone for data grab

        timestamp_new = pd.to_datetime(
            timestamp, format='%Y-%m-%d', errors='coerce')
        # end_new = pd.to_datetime(end, format='%Y-%m-%d', errors='coerce') #Changing to datetime format so can convert to local time

        timestamp_new = timestamp_new.tz_localize(
            local_zone)  # .tz_convert(pacific)
        # end_new = end_new.tz_localize('America/Los_Angeles')# pytz.utc .tz_convert(pacific) # Localizing times so request reflects PT time and not utc

        #start_new = start_new.tz_localize(None)
        #end_new = end_new.tz_localize(None)

        timestamp_new = timestamp_new.strftime('%Y-%m-%d %H:%M:%S')
        # end_new = end_new.strftime('%Y-%m-%d %H:%M:%S') # Converting datetime back to string for get request

        return timestamp_new  # , end_new

    """

    The following section contains methods to compose the url and parameters for server requests

    """

    def _compose_stream_url(self, Web_ID,
                            start,
                            end,
                            calculation,
                            interval,
                            sumType,
                            summaryDuration,
                            label):
        """
        This method composes the url to get the stream
        """
        # constructs the first part of the http call
        Web_ID_string = self.root + "streams/" + Web_ID + "/" +  calculation

        # time conversions
        if start != "t" and start != "y" and start != "*":
            start = self._local_to_utc(start, 'America/Los_Angeles')

        if end != "t" and end != "y" and end != "*":
            # adjusting time because requested timeframe is based on utc time
           end = self._local_to_utc(end, 'America/Los_Angeles')

        # constructs the parameters for requests REST api call
        if  calculation == "summary":
            parms = {"startTime": start, "endTime": end,
                     "summaryDuration": summaryDuration, "summaryType":sumType,
                     "selectedFields": "Items.Type;Items.Value.Timestamp;Items.Value.Value"
                    }
        else:
            parms = {"startTime": start, "endTime":end,
                     "interval": interval, "maxCount": 100000,
                     "selectedFields": "Items.Timestamp;Items.Value"
                    }

        return Web_ID_string, parms

    def _compose_point_search_url(self, 
                            dataserver,
                            point_query,
                            include_Path
                            ):

        """
        This method composes the url to get the point names and WebIDs
        """
        point_search = "/points?nameFilter="
        query_string = self.root + "dataservers/" + \
            dataserver + point_search + point_query

        if include_Path: # this includes the path in the output

            parms = {"startIndex": 0, "maxCount": 100000,
                 "selectedFields": "Items.WebId;Items.Name;Items.Path"}
        
        else: # this does not include Path

            parms = {"startIndex": 0, "maxCount": 100000,
                     "selectedFields": "Items.WebId;Items.Name"}

        return query_string, parms

    def _compose_path_search_url(self,
                            path_query,
                            include_Path
                            ):

        """
        This method composes the url to get the point names and WebIDs
        """

        # when searching with a path, need to include this part. Only real difference from search_by_point
        path_search = "attributes?path=\\"
        # query_string does not need dataserver.
        query_string = self.root + path_search + path_query

        #        print query_string # "//' needs extra for escape character or else will not evaluate correctly.

        # Json no longer need items, because single item returned. Different from search_by_point

        if include_Path: # this includes the path in the output

            parms = {"startIndex": 0, "maxCount": 100000,
                 "selectedFields": "WebId;Name;Path"}
        
        else: # this does not include Path

            parms = {"startIndex": 0, "maxCount": 100000,
                     "selectedFields": "WebId;Name"}

        return query_string, parms

    def search_by_point(self,
                        point_query,
                        dataserver="s09KoOKByvc0-uxyvoTV1UfQVVRJTC1QSS1Q",
#                        include_WebID=True,
                        include_Path=False
                        ):
        """ 
        This method searches for points allowing * operators. It returns point list and a Dictionary with names:WebIDs

        arguments:

        point_query: point name expression (allows *)
        dataserver: default point to UC Davis
        include_WebID: by default True, ut returns list AND a dictionary {name : Web_ID, ...}


        Example API json returned:

        {
          "Items": [
            {
              "WebId": "P09KoOKByvc0-uxyvoTV1UfQ61oCAAVVRJTC1QSS1QXFBFUy5BSFUuQ09PTElORyBFTkVSR1kgQlRVIFBFUiBIUg",
              "Name": "PES.AHU.Cooling Energy BTU per Hr"
            },
            {
              "WebId": "P09KoOKByvc0-uxyvoTV1UfQ7FoCAAVVRJTC1QSS1QXFBFUy5BSFUuSEVBVElORyBFTkVSR1kgQlRVIFBFUiBIUg",
              "Name": "PES.AHU.Heating Energy BTU per Hr"
            }
        }

        returns:
        It returns a list with point names and a dictionary with name : Web_ID

        """

        # compose url
        query_string, parms = self._compose_point_search_url(dataserver, point_query, include_Path)
        

        try:
            # call request library
            response = req.get(query_string, parms)

            # to manage errors
            response.raise_for_status()
            
            # parse result and return it
            return self._parse_point(response, include_Path)
    
        except req.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            return [], {}
        except req.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            return [], {}
        except req.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
            return [], {}
        except req.exceptions.RequestException as err:
            print ("OOps: Something Else Happened",err)
            return [], {}

    def search_by_path(self,
                       path_query,
                       include_Path=False,
                       ):
        """ 
        This method searches for path allowing * operators. It returns path list and a Dictionary with paths:WebIDs

        arguments:

        path_query: point name expression (allows *)
        include_WebID: by default True, ut returns list AND a dictionary {name : Web_ID, ...}


        Example API json returned:


        {
          "WebId": "A0EbgZy4oKQ9kiBiZJTW7eugwS5GAMtE55BGIPhgDcyrprwcrOde7rrSVobodgP17EChQVVRJTC1BRlxDRUZTXFVDREFWSVNcQlVJTERJTkdTXFJJRkxFIFJBTkdFXEVMRUNUUklDSVRZfERFTUFORF9LQlRV",
          "Name": "Demand_kBtu"
        }

        NOTE: this is different from the JSON obtained by search_by_point

        returns:
        It returns a list with point names and a dictionary with name : Web_ID

        """

        # compose url
        # when searching with a path, need to include this part. Only real difference from search_by_point
        #path_search = "attributes?path=\\"
        # query_string does not need dataserver.
        #query_string = self.root + path_search + path_query

        #        print query_string # "//' needs extra for escape character or else will not evaluate correctly.

        
        # Json no longer need items, because single item returned. Different from search_by_point
        #parms = {"startIndex": 0, "maxCount": 100000,
        #         "selectedFields": "WebId;Name"}
        
        query_string, parms = self._compose_path_search_url(path_query, include_Path)


        try:
            # call requests library
            response = req.get(query_string, parms)
            # to manage errors
            response.raise_for_status()
            #print response  # Should be getting a Response of 200 if successful
            # parse result and return it
            return self._parse_path(response, include_Path)
    
        except req.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            return [], {}
        except req.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            return [], {}
        except req.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
            return [], {}
        except req.exceptions.RequestException as err:
            print ("OOps: Something Else Happened",err)
            return [], {}


    def get_webID_by_point(self,
                           point_name,                           
                           dataserver="s09KoOKByvc0-uxyvoTV1UfQVVRJTC1QSS1Q" # defaults to buildings dataserver
                           ):
        """
        This method is to make sure we get a single WebID as result of the get_stream_by_point search

        """

        pointList, PointDic = self.search_by_point(
            point_name)

        if len(pointList) > 1:
            print ("warining: the query returned more than one WebID n=%d, \
            only the first one is used\n returning only first" % len(pointList))
        
        try:
            Web_ID_ = PointDic[pointList[0]]["WebId"]

        except:
            Web_ID_ = None

        return Web_ID_

    def get_webID_by_path(self,
                          path_name,
                          ):
        """
        This method is to make sure we get a single WebID as result of the get_stream_by_path search

        """

        pointList, PointDic = self.search_by_path(
            path_name)  # finding webId with path name

        if len(pointList) > 1:
            print ("warning: the query returned more than one WebID n=%d, \
            only the first one is used\n returning only first" % len(pointList))

        try:
            Web_ID_ = PointDic[pointList[0]]["WebId"]

        except:
            Web_ID_ = None

        return Web_ID_

    def _get_stream(self, Web_ID=None,
                   start="y",
                   end="t",
                   calculation="calculated",
                   interval="1h",
                   sumType="All",
                   label=None):
        """ 
        This method gets the stream given a WebID. It works with one stream at the time.

        arguments: 
        Web_ID=None : - the unique identifier of the time series 
        start="y" : - start date, default yesterday "y"; can use different formats as "YYYY-MM-DD";
       end="t" : - end date, default yesterday "t"; can use different formats as "YYYY-MM-DD";        
         calculation="interpolated": can use "recorded" to get raw data and summary to get summary data (tot, mean, sd);
        note: summary data is not a time series, but a dictionary
        interval="1h": interpolation interval, used only with interpolated; default 1 hour
       sumType="All" : used if calculation is "summary", can use All, Total, default All
        label=None : used to pass around name of the column in the dataframe or can overwrite it

        returns:
        DataFrame object for TS
        dictionary for summary
        single value for end

        """
        summaryDuration = None
        
        # create a function similar to the "calculated" function in PI_datalink
        if  calculation == "calculated":            
            
            calculation = "summary"
            summaryDuration = interval
            if sumType == "All":
               sumType = "Average"
            
        # call function that constructs the http call for method /streams (see PI API manual for API details)
        Web_ID_string, parms = self._compose_stream_url(
                Web_ID, start, end,  calculation, interval,sumType, summaryDuration, label)

        

        # manage exceptions

        try:
            # call python library for json/http REST API
            response = req.get(Web_ID_string, parms)
            # to manage exceptions
            response.raise_for_status()

            # prints for testing: <Response [200]> means it works
            print (response)

            # parse the response
            if ( calculation == ("interpolated")) | ( calculation == ("recorded")):
                result = self._parse_TS(response, Web_ID, label)
            elif (summaryDuration):
                if ( calculation == ("summary")):
                    result = self._parse_summary_with_duration(response, Web_ID, label)
            elif  calculation == ("summary"):
                result = self._parse_summary(response, Web_ID, label)         
            elif  calculation == ("end"):
                result = self._parse_end(response, Web_ID, label)
            
            return result

        except req.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            return pd.DataFrame()
        except req.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            return pd.DataFrame()
        except req.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
            return pd.DataFrame()
        except req.exceptions.RequestException as err:
            print ("OOps: Something Else Happened",err)
            return pd.DataFrame()


    def get_stream_by_point(self,
                            point_names,
                            start="y",
                            end="t",
                            calculation="calculated",
                            interval="1h",
                            sumType="All",
                            label=None,
                            dataserver="s09KoOKByvc0-uxyvoTV1UfQVVRJTC1QSS1Q", # defaults to UCD buildings dataserver
                            WebID_dic=None
                            ):
        """ 
        This method gets the stream given a the point name. 
        It calls get_webID_by_point to get a single Web ID by point name
        Then it calls the stream using the Web ID
        It also works with multiple points, but it is not optimized (can save time by calling batches)

        arguments: 
        point_names : name or list of PI point names
        start="y" : - start date, default yesterday "y"; can use different formats as "YYYY-MM-DD";
        end="t" : - end date, default yesterday "t"; can use different formats as "YYYY-MM-DD";        
        calculation="interpolated": can use "recorded" to get raw data and summary to get summary data (tot, mean, sd);
        note: summary data is not a time series, but a dictionary
        interval="1h": interpolation interval, used only with interpolated; default 1 hour
        sumType=All : used if calculation is "summary", can use All, Total, default All
        label=None : used to pass around name of the column in the dataframe or can overwrite it


        returns:
        DataFrame object for TS
        dictionary for summary
        single value for end

        """
        # if point_names is a list, downloas all of them ,then merges them into a dataframe

        # case 1: multiple streams
        if isinstance(point_names, list):

            streams_df = pd.DataFrame()

            for point_name in point_names:

                # get webID from point name
                Web_ID = self.get_webID_by_point(point_name, dataserver)

                if Web_ID:

                    stream = self._get_stream(
                        Web_ID, start,end,  calculation, interval,sumType, label=point_name)
                
                    if streams_df.empty:
                        streams_df = pd.DataFrame(stream)
                    else:
                        streams_df = streams_df.join(stream, how="outer")
                else:
                    print(point_name + "not found")

            return streams_df

        # case 2: single stream
        else:

            Web_ID = self.get_webID_by_point(point_names, dataserver)

            if Web_ID:
                stream = self._get_stream(
                    Web_ID, start,end,  calculation, interval,sumType, label=point_names)

            else:
                print(point_names + "not found")
                stream = pd.DataFrame()

        return stream


    def get_stream_by_path(self,
                           path_names,
                           start="y",
                           end="t",
                           calculation="calculated",
                           interval="1h",
                           sumType="All",
                           label=None,
                           WebID_dic=None
                           ):
        """ 
        This method gets the stream given a the the path.
        Since the path is the key of the database the call to the API does not use the dataserver as before (points names are are unique only on a dataserver) -> the url composed is a bit different 
        It calls get_webID_by_path to get a single Web ID by path
        Then it calls the stream using the Web ID
        It also works with multiple paths, but it is not optimized (can save time by calling batches)

        arguments: 
        path_names : name or list of PI paths
        start="y" : - start date, default yesterday "y"; can use different formats as "YYYY-MM-DD";
       end="t" : - end date, default yesterday "t"; can use different formats as "YYYY-MM-DD";        
         calculation="interpolated": can use "recorded" to get raw data and summary to get summary data (tot, mean, sd);
        note: summary data is not a time series, but a dictionary
        interval="1h": interpolation interval, used only with interpolated; default 1 hour
       sumType=All : used if calculation is "summary", can use All, Total, default All
        label=None : used to pass around name of the column in the dataframe or can overwrite it

        returns:
        DataFrame object for TS
        dictionary for summary
        single value for end

        """

        # if send a list of points downloads all of them then merges it into a dataframe
        # Same as get_stream_by_point except for variable names and values passed into function calls.

        # case 1: multiple streams
        if isinstance(path_names, list):

            streams_df = pd.DataFrame()

            for path_name in path_names:

                # get webID with given path name
                Web_ID = self.get_webID_by_path(path_name)

                if Web_ID:

                    stream = self._get_stream(
                        Web_ID, start,end,  calculation, interval,sumType, label=path_name)
                
                    if streams_df.empty:
                        streams_df = pd.DataFrame(stream)
                    else:
                        streams_df = streams_df.join(stream, how="outer")
                else:
                    print(path_name + "not found")

            return streams_df

        # case 2: single stream
        else:

            Web_ID = self.get_webID_by_path(path_names)

            if Web_ID:
                stream = self._get_stream(
                    Web_ID, start,end,  calculation, interval,sumType, label=path_names)

            else:
                print(path_names + "not found")
                stream = pd.DataFrame()

        return stream
