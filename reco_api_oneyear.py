from flask import Flask,request
from flask_restful import reqparse, abort, Api, Resource
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from pprint import pprint
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.cluster import KMeans
import json
from collections import OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import geocoder
import random
from time import sleep
from sklearn.externals import joblib
from datetime import datetime
import warnings
from sqlalchemy import create_engine
import pymysql

#Ignore warnings
warnings.filterwarnings('ignore')

#connect to mysql
db = pymysql.connect("localhost", "root", "0702","mytest")
cnx = create_engine('mysql+pymysql://root:0702@localhost/mytest',echo=False)
cursor = db.cursor()

############ all active drivers dictionary
active_driver_dict = {
                    "B8":{"phone":"347-248-4291", "name":"Zhong Wei",},
                    "B9":{"phone":"917-216-6039","name":"Jin Jing Jie"},
                    "B24":{"phone":"347-491-9959", "name":"Yu Fulin"},
                    "B25":{"phone": "646-371-0604", "name": "Ismael Sierra Balbuena"},
                    "B26":{"phone":"917-388-5184", "name":"Li Lie"},
                    "B35":{"phone":"917-257-1881", "name":"Li Chong Yuet"},
                    "B36":{"phone":"347-925-7791", "name":"Zhao Huaibao"},
                    "B37":{"phone":"917-769-2585", "name":"Chang Chao Lung"},
                    "B38":{"phone":"718-309-3562","name":"Wen Woei Wong"},
                    "B42":{"phone":"646-286-1384", "name":"Hu Jing"},
                    "B62":{"phone":"917-607-8928", "name":"Lin Zhengang"},
                    "B68":{"phone":"646-675-5476", "name":"Chan Kam Fong"},
                    "B88":{"phone":"917-567-0170", "name":"Yung Sammy"},
                    "S5":{"phone":"917-861-5898", "name":"Chen Junzhi"},
                    "S18":{"phone":"917-225-2849", "name":"Morris Lam"},
                    "S19":{"phone":"347-513-6065", "name":"Cabarcas Gabriel"},
                    "S21":{"phone":"917-216-5839", "name":"Liao Han Chieh"},
                    "S22":{"phone":"929-377-9999","name":"Yu Xiaojing"},
                    "S26":{"name":"Sahid Carias", "phone":"646-258-5118"},
                    "S27":{"name":"Sun Zhenkai", "phone":"917-257-2639"},
                    "S29":{"name":"Lam Chuen","phone":"646-705-9113"},
                    "S30":{"name":"Guo Zhongguang", "phone":"347-925-2065"},
                    "S31":{"name":"Zhang Ming", "phone":"646-203-2678"},
                    "S32":{"name":"Michael Li", "phone":"718-331-7188"},
                    "S35":{"name":"Nie Bi Qiu", "phone":"917-480-9482"},
                    "S37":{"name":"Man Kam Loi", "phone":"646-797-1731"},
                    "S39":{"name":"Chen Jun Yi", "phone":"646-525-1200"},
                    "S41":{"name":"Michael Wu", "phone":"347-280-6988"},
                    "S42":{"name":"Emmanuel Valdez","phone":"347-752-0713"},
                    "S45":{"name":"Ding Guo Yuan","phone":"917-530-7839"},
                    "S46":{"name":"Zhao Wei Cheng Paul", "phone":"917-293-9981"},
                    "S47":{"name":"Zhang Guiquan","phone":"646-326-1269"},
                    "S49":{"name":"Clinton Steer Solis","phone":"347-822-0931"},
                    "S50":{"name":"Zhu Gensheng","phone":"718-219-9897"},
                    "S66":{"name":"Lu Vivian","phone":"718-887-1282"},
                    "S68":{"name":"Li Hua Yong", "phone":"917-295-6577"},
                    "S70":{"name":"Tina Chen", "phone":"917-582-5278"},
                    "S74":{"name":"Jian-Wu Qi","phone":"347-988-2121"},
                    "S79":{"name":"Tam Chi","phone":"917-601-3800"},
                    "S86":{"name":"Dai Xian Zao","phone":"718-795-3032"},
                    "S98":{"name":"Lee YihHer","phone":"917-838-5618"},
                    "B12":{"name":"Law Chi (Jack)","phone":"516-884-4828"},
                    "G6": {"name":"Mohammad Rahman","phone":"914-258-6609"}
                    }

app = Flask(__name__)
api = Api(app)

########## All post data arguments
parser = reqparse.RequestParser()
parser.add_argument('baseId')
parser.add_argument('agencyId')
parser.add_argument("lastName")
parser.add_argument("firstName")
parser.add_argument('pickupDate')
parser.add_argument('pickupTime')
parser.add_argument('pickupAddress')
parser.add_argument('dropOffAddress')
parser.add_argument('pickupCity')
parser.add_argument('dropOffCity')
parser.add_argument('dropOffZipCode')
parser.add_argument('pickupPoint')
parser.add_argument('dropOffPoint')

######## clean address by removing words like APT, FL
def clean_address(address):
    temp = address.split(" ")
    for i in temp:
        if i in ['AV','AVE','AVENUE','BLD','BOULEVARD', 'BLDG','BOWERY','BROADWAY','CI',
                 'CT','CIR','DR', 'DRIVER', 'EXP','EXPY', 'EXPRESSWAY', 'EXPWY','EXWY',
                 'HIGHWAY','HWY','LN','PL','PI','PARKWAY','PLACE','PLZ','PKWY','RD','ROAD',
                 'SQ','STR','ST','STREET','SQUARE','TNPK','TPKE','TURNPIKE','WAY','MALL']:
            del temp[temp.index(i)+1:]

    return " ".join(temp)


######## get geolocation
def get_geolocation(address):
    try:
        temp = geocoder.arcgis(address).latlng
        if not 40<temp[0]<41.5 or not -74.0<temp[1]<-73.0 or temp is None:
            cache = [[40.746725, -73.826921], [40.717196, -73.998893], [40.754184, -73.984531],[40.780733, -73.958817],
                    [40.762906, -73.924716], [40.719805, -73.910258],[40.676604, -73.920870], [40.785070, -73.839912]]
            temp = random.choice(cache)
        sleep(0.05)

    except:
        temp = geocoder.google(address).latlng
        sleep(0.05)

    return temp

pickup_data = []
dropoff_data = []

class PreProcess():
    def __init__(self):
        pass

    def show_head(self):
        return self.df.head()

    def load(self,file):
        self.df = pd.read_csv(file, sep=',')
        return self.df

    def process(self,file):
         pass

    def get_features(self):
        SQL = """select company_id, customer_id, date_day_id,day_week_id,time_id,roundtrip_city_id,roundtrip_loc_id from oneyear"""
        self.X = pd.read_sql(SQL, cnx)
        return self.X

    def get_targets(self):
        SQL = """select fleet_id from oneyear"""
        self.Y = pd.read_sql(SQL,cnx)
        return self.Y

# Load data from mysql databse
proc = PreProcess()
X = proc.get_features()
Y = proc.get_targets()

# Load k-means models from local
kmeans_pickup = joblib.load('kmeans_pickup_model_oneyear.sav')
kmeans_dropoff = joblib.load('kmeans_dropoff_model_oneyear.sav')



#initialize PCA

# pca = PCA(n_components=2, svd_solver='auto',tol=1.0,iterated_power=10,whiten=True,random_state=40).fit(X_train)
# pca_model_file = 'pca_model_oneyear.sav'
# joblib.dump(pca,pca_model_file)

pca = joblib.load('pca_model_oneyear.sav')

#transfer the original data to pca form

# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)

#######  initialize KNN classifer   ##########
# knn_clf = KNeighborsClassifier(n_neighbors=10, weights='distance',algorithm='auto', leaf_size=30, p=1).fit(X_train_pca,Y_train)
# knn_model_file = 'knn_model_oneyear.sav'
# joblib.dump(knn_clf,knn_model_file)

knn_clf = joblib.load('knn_model_oneyear.sav')
# print(knn_clf.score(X_test_pca,Y_test)*100)

class trans_rawdata_to_id():
    def __init__(self):
        pass

    def get_company_id(self,company):
        try:
            cursor.execute("select DISTINCT company_id from oneyear WHERE company=%s",company)
            self.company_id = cursor.fetchone()[0]
        except:
            self.company_id = -1
        return self.company_id

    def get_customer_id(self,cust_fullname):
        try:
            SQL = "select DISTINCT customer_id from oneyear WHERE cust_fullname=" + "'" + cust_fullname + "'"
            cursor.execute(SQL)
            self.customer_id = cursor.fetchone()[0]
        except:
            self.customer_id = -1
        return self.customer_id

    def get_date_id(self,date):
        try:
            SQL = "select DISTINCT date_id from oneyear WHERE date=" + "'" + date + "'"
            cursor.execute(SQL)
            self.date_id = cursor.fetchone()[0]
        except:
            self.date_id = -1
        return self.date_id

    def get_day_week_id(self,day_week):
        try:
            SQL = "select DISTINCT day_week_id from oneyear WHERE day_of_week=" + str(day_week)
            cursor.execute(SQL)
            self.day_week_id = cursor.fetchone()[0]
        except:
            self.day_week_id = -1
        return self.day_week_id

    def get_time_id(self,time):
        try:
            SQL = "select DISTINCT time_id from oneyear WHERE time=" + "'" + time + "'"
            cursor.execute(SQL)
            self.time_id = cursor.fetchone()[0]
        except:
            self.time_id = -1
        return self.time_id

    def get_pickup_loc_id(self, pickup_loc):
        try:
            self.pickup_geo_kmeans_id = int(kmeans_pickup.predict([pickup_loc]))
        except:
            self.pickup_geo_kmeans_id = -1
        return self.pickup_geo_kmeans_id

    def get_dropoff_loc_id(self,dropoff_loc):
        try:
            self.dropoff_geo_kmeans_id = int(kmeans_dropoff.predict([dropoff_loc]))
        except:
            self.dropoff_geo_kmeans_id = -1

        return self.dropoff_geo_kmeans_id

    def get_roundtrip_loc_id(self,pickup_loc,dropoff_loc):
        try:
            SQL = "select DISTINCT roundtrip_loc_id from oneyear WHERE pickup_location_id="+str(self.get_pickup_loc_id(pickup_loc))+" and dropoff_add_id="+str(self.get_dropoff_loc_id(dropoff_loc))
            cursor.execute(SQL)
            self.roundtrip_loc_id = cursor.fetchone()[0]
        except:
            self.roundtrip_loc_id = -1
        return self.roundtrip_loc_id

    def get_roundtrip_city_id(self,pickup_city, dropoff_city):
        try:
            SQL = "select DISTINCT roundtrip_city_id from oneyear WHERE pickup_city="+"'"+pickup_city+"'"+" and dropoff_city="+"'"+dropoff_city+"'"
            cursor.execute(SQL)
            self.roundtrip_city_id = cursor.fetchone()[0]
        except:
            self.roundtrip_city_id = -1
        return self.roundtrip_city_id

    def to_id(self):
        self.id_data = [[self.company_id,self.customer_id,self.date_id,self.day_week_id,self.time_id,self.roundtrip_loc_id,self.roundtrip_city_id]]
        return self.id_data


class Classifier():
    def __init__(self):
        pass

    def pred_1st(self,pca):
        self.the_1st = knn_clf.predict(pca)
        return self.the_1st[0]

    def pred_the_rest(self,pca,n):
        result = []

        self.the_rest = knn_clf.kneighbors(pca,n_neighbors=n,return_distance=False)
        cursor.execute("select distinct fleet_id from oneyear where unique_id in " + str(tuple(self.the_rest[0])))
        temp = cursor.fetchall()
        cache = list(map(lambda x:result.append(x[0]), temp))

        if self.the_1st[0] in result:
            result.remove(self.the_1st[0])

        result = list(set(result))
        return result

    def get_driver_FN(self,id):
        SQL = "select driver_FN from oneyear WHERE fleet_id="+str(id)
        cursor.execute(SQL)
        self.driver_FN = cursor.fetchone()[0]
        return self.driver_FN

    def get_driver_LN(self,id):
        SQL = "select driver_LN from oneyear WHERE fleet_id="+str(id)
        cursor.execute(SQL)
        self.driver_LN = cursor.fetchone()[0]
        return self.driver_LN

    def get_driver_phone(self,id):
        SQL = "select driver_phone from oneyear WHERE fleet_id=" + str(id)
        cursor.execute(SQL)
        self.driver_phone = cursor.fetchone()[0]
        return self.driver_phone

    def get_driver_fleet(self,id):
        SQL = "select fleet from oneyear WHERE fleet_id=" + str(id)
        cursor.execute(SQL)
        self.driver_fleet = cursor.fetchone()[0]
        return self.driver_fleet

    def get_textdata(self,id):
        SQL = "select DISTINCT driver_FN, driver_LN, driver_phone,fleet from oneyear where fleet_id IN"+str(id)
        cursor.execute(SQL)
        temp = cursor.fetchall()

        self.driver_FN = list(map(lambda x:x[0], set(temp)))
        self.driver_LN = list(map(lambda x: x[1], set(temp)))
        self.driver_phone = list(map(lambda x: x[2], set(temp)))
        self.driver_fleet = list(map(lambda x: x[3], set(temp)))
        return self.driver_FN,self.driver_LN,self.driver_phone,self.driver_fleet

    def get_1st_data(self,id):
        SQL = "select DISTINCT driver_FN, driver_LN, driver_phone,fleet from oneyear where fleet_id=" + str(id)
        cursor.execute(SQL)
        temp = cursor.fetchone()
        self.driver_FN = temp[0]
        self.driver_LN = temp[1]
        self.driver_phone = temp[2]
        self.driver_fleet = temp[3]
        return self.driver_FN, self.driver_LN, self.driver_phone, self.driver_fleet

    def get_trip_number(self,id,cust_id):

        SQL = "select fleet,count(fleet) as fleet_count from oneyear WHERE fleet IN "+str(id)+" and customer_id="+str(cust_id)+" GROUP BY fleet order by fleet_count desc"
        cursor.execute(SQL)
        temp = cursor.fetchall()

        return temp

    def get_trip_number2(self,fleet,fleet_id,cust_id):
        temp = self.get_trip_number(fleet_id,cust_id)
        for i in temp:
            if fleet==i[0]:
                return i[1]
            else:return 0

    def get_trip_number_1st(self,id,cust_id):
        # start = datetime.now()
        SQL = "select count(fleet) from oneyear where fleet="+str(id)+" and customer_id="+str(cust_id)
        cursor.execute(SQL)
        temp = cursor.fetchall()
        # print(datetime.now() - start)
        return temp[0][0]

my_dict = OrderedDict()
res_dict = OrderedDict()

class my_api(Resource):

    def get(self):
        return res_dict

    def post(self):
        args = parser.parse_args()
        trans = trans_rawdata_to_id()


        # Save post data into a temp dictionary, then we can use the data from the dict
        my_dict['user info'] = {'base':args['baseId'],'company':args['agencyId'].upper(),'cust_FN': args["firstName"].upper(),'cust_LN':args['lastName'].upper(),
                   'date':args['pickupDate'],'time':args['pickupTime'], 'pickup location':args['pickupAddress'].upper(),
                   'pickup city': args['pickupCity'].upper(),'dropoff location': args['dropOffAddress'].upper(),
                   'dropoff city':args['dropOffCity'].upper(),'pickup point':args['pickupPoint'], 'dropoff point':args['dropOffPoint']}

        ######## Transfer post data into pandas id format
        company_id = trans.get_company_id(my_dict['user info']['company'])
        cust_fullname = my_dict['user info']['cust_FN']+" "+my_dict['user info']['cust_LN']
        cust_fullname_id = trans.get_customer_id(cust_fullname)
        cache_time = my_dict['user info']['date']
        try:
            cache_time = datetime.strptime(cache_time, '%m/%d/%y')
        except:
            cache_time = datetime.strptime(cache_time, '%m/%d/%Y')

        date_day_id = cache_time.day

        day_week = cache_time.isoweekday()

        day_week_id = trans.get_day_week_id(day_week)

        time_id = trans.get_time_id(my_dict['user info']['time'])


        roundtrip_loc_id = trans.get_roundtrip_loc_id(my_dict['user info']['pickup point'], my_dict['user info']['dropoff point'])

        roundtrip_city_id = trans.get_roundtrip_city_id(my_dict['user info']['pickup city'], my_dict['user info']['dropoff city'])


        # Merge all data into a list, so that pca can transform it
        id_data = [[company_id, cust_fullname_id,date_day_id,day_week_id,time_id,roundtrip_loc_id,roundtrip_city_id]]

        #PCA transforms data
        to_pca = pca.transform(id_data)

        #New a classifier
        classifier = Classifier()

        #Use classifier to predict the 1st driver.
        #Note that the result is the index, not the exact fleet number of driver.

        the_1st = classifier.pred_1st(to_pca)

        #Use classifier to predict the rest of drivers.
        the_rest = classifier.pred_the_rest(to_pca,11)

        temp = []

        ######### transfer index number into text information for the 1st driver
        temp_1st_text = classifier.get_1st_data(the_1st)
        temp.append({'firstName':temp_1st_text[0], 'lastName':temp_1st_text[1],
                     'phone':temp_1st_text[2], 'fleetNum':temp_1st_text[3],'priority':'1',
                     'NumberOfTrips':classifier.get_trip_number_1st(the_1st,cust_fullname_id)})


        ######## transfer index number into text information for the rest of drivers
        j = 2
        tuple_the_rest = tuple(the_rest[0:9])
        # print(tuple_the_rest)
        result_the_rest = classifier.get_textdata(tuple_the_rest)

        def get_trip_number_dict(id, cust_id):
            dict_temp = {key: 0 for key in result_the_rest[3]}
            SQL = "select fleet,count(fleet) as fleet_count from oneyear WHERE fleet_id IN " + str(id) + " and customer_id=" + str(cust_id) + " GROUP BY fleet order by fleet_count desc"
            cursor.execute(SQL)
            temp = cursor.fetchall()

            for i in temp:
                dict_temp[i[0]] = i[1]


            return dict_temp

        dict_temp = get_trip_number_dict(tuple_the_rest, cust_fullname_id)


        for i in range(9):
            temp.append({'firstName': result_the_rest[0][i], 'lastName':result_the_rest[1][i] ,
                         'phone':result_the_rest[2][i], 'fleetNum':result_the_rest[3][i], 'priority':str(j),
                         'NumberOfTrips':dict_temp[result_the_rest[3][i]]})
            j+=1

        ###########Restore all drivers' info in to the dictionary, then return it.
        res_dict['data'] = temp
        res_dict["result"] = "SUCCESS"

        return res_dict

api.add_resource(my_api, '/')

##### enterance of the program
if __name__=='__main__':
    app.run(debug=False,host='127.0.0.1', port='8888')
