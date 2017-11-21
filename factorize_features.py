import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import geocoder
import random
from time import sleep
from sklearn.externals import joblib
from datetime import datetime

pickup_data = []
dropoff_data = []

# Remove all text after the words in the list
def clean_address(x):
    a = x.replace(".", " ").split(" ")
    for i in a:
        if i in ['AV','AVE','AVENUE','BLD','BOULEVARD','BLVD', 'BLDG','BOWERY','BROADWAY','CI',
                 'CT','CIR','DR', 'DRIVER', 'EXP','EXPY', 'EXPRESSWAY', 'EXPWY','EXWY',
                 'HIGHWAY','HWY','LN','PL','PI','PARKWAY','PLACE','PLZ','PKWY','RD','ROAD',
                 'SQ','STR','ST','STREET','SQUARE','TNPK','TPKE','TURNPIKE','WAY','MALL']:
            del a[a.index(i)+1:]
    return " ".join(a)

# Get Geo-location points
def get_geolocation(address):
    try:
        temp = geocoder.arcgis(address).latlng
        if not 40<temp[0]<41.5 or not -74.0<temp[1]<-73.0 or temp is None:
            cache = [[40.746725, -73.826921], [40.717196, -73.998893], [40.754184, -73.984531],[40.780733, -73.958817],
                    [40.762906, -73.924716], [40.719805, -73.910258],[40.676604, -73.920870], [40.785070, -73.839912]]
            temp = random.choice(cache)
        sleep(0.02)

    except:
        temp = geocoder.google(address).latlng
        sleep(0.05)
    print(temp)
    return temp

df = pd.read_csv('merged_one_year.csv',encoding='ISO-8859-1')
df = df.dropna(how='any') # Drop any NaN records


df2 = df.apply(lambda x:x.astype(str).str.upper()) # Captialize all records

df2['driver_phone'] = df2['driver_phone'].apply(lambda x: x.replace("-", ""))
df2['driver_name'] = df2['driver_name'].apply(lambda x: x.split(" "))
df2['driver_LN'] = df2['driver_name'].apply(lambda x: x[0])
df2['driver_FN'] = df2['driver_name'].apply(lambda x: "".join(x[1:]))
df2['date_day'] = df2['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').day)
df2['day_of_week'] = df2['date'].apply(lambda x: datetime.strptime(x,'%m/%d/%y').isoweekday())

df2['company_id'] = pd.factorize(df2.company)[0]
df2['date_id'] = pd.factorize(df2.date)[0]
df2['date_day_id'] = pd.factorize(df2.date_day)[0]
df2['day_week_id'] = pd.factorize(df2.day_of_week)[0]
df2['dropoff_city_id'] = pd.factorize(df2.dropoff_city)[0]
df2['fleet_id'] = pd.factorize(df2.fleet)[0]
df2['pickup_city_id'] = pd.factorize(df2.pickup_city)[0]
df2['time_id'] = pd.factorize(df2.time)[0]
df2['roundtrip_city_id'] = pd.factorize(df2.pickup_city + df2.dropoff_city)[0]
df2['cust_fullname'] = df2['cust_FN'] + " " + df2['cust_LN']
df2['customer_id'] = pd.factorize(df2.cust_fullname)[0]


df2['cleaned_pickup_location'] = df2['pickup_location'].apply(lambda x:clean_address(x)+" NY")
df2['cleaned_dropoff_location'] = df2['dropoff_location'].apply(lambda x:clean_address(x)+" NY")

df2['latlng_pickup_location'] = df2['cleaned_pickup_location'].apply(lambda x:get_geolocation(x))
df2['latlng_dropoff_location'] = df2['cleaned_dropoff_location'].apply(lambda x:get_geolocation(x))

df2['latlng_pickup_location'].apply(lambda x: pickup_data.append([float(x[0]), float(x[1])]))
df2['latlng_dropoff_location'].apply(lambda x: dropoff_data.append([float(x[0]), float(x[1])]))

df2['latlng_pickup_cleaned'] = pickup_data
df2['latlng_dropoff_cleaned'] = dropoff_data


kmeans_pickup = KMeans(n_clusters=30, init='random', precompute_distances=True, random_state=42,
                       algorithm='auto').fit(pickup_data)
kmeans_dropoff = KMeans(n_clusters=30, init='random', precompute_distances=True, random_state=42,
                        algorithm='auto').fit(dropoff_data)

kmeans_pickup_file = 'kmeans_pickup_model_oneyear.sav'
kmeans_dropoff_file = 'kmeans_dropoff_model_oneyear.sav'
joblib.dump(kmeans_pickup, kmeans_pickup_file)
joblib.dump(kmeans_dropoff, kmeans_dropoff_file)

df2['pickup_location_id'] = df2['latlng_pickup_cleaned'].apply(lambda x: int(kmeans_pickup.predict(np.array([x]))))
df2['dropoff_location_id'] = df2['latlng_dropoff_cleaned'].apply(lambda x: int(kmeans_dropoff.predict(np.array([x]))))

df2['roundtrip_loc_id'] = pd.factorize(df2.pickup_location_id + df2.dropoff_location_id)[0]

df2.to_csv('merged_one_year_part1.csv')
