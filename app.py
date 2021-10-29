#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dash import dcc
from dash import html
import dash_daq
from flask import Flask
import os
import base64
import io
from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, State, Trigger
from dash.exceptions import PreventUpdate
import plotly.express as px
import json
import random
import dash_bootstrap_components as dbc
from sklearn.metrics import silhouette_score
import time
from datetime import datetime
import io



# Käytetyt värit.
# https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F
colors = pd.read_csv('colors_wikipedia.csv')
colors.index+=1
colors.index = colors.index.astype(int)

url = 'https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/Kuntien_avainluvut/2021/kuntien_avainluvut_2021_viimeisin.px'

kunta_payload = """

{
  "query": [
    {
      "code": "Alue 2021",
      "selection": {
        "filter": "item",
        "values": [
          "SSS",
          "020",
          "005",
          "009",
          "010",
          "016",
          "018",
          "019",
          "035",
          "043",
          "046",
          "047",
          "049",
          "050",
          "051",
          "052",
          "060",
          "061",
          "062",
          "065",
          "069",
          "071",
          "072",
          "074",
          "075",
          "076",
          "077",
          "078",
          "079",
          "081",
          "082",
          "086",
          "111",
          "090",
          "091",
          "097",
          "098",
          "102",
          "103",
          "105",
          "106",
          "108",
          "109",
          "139",
          "140",
          "142",
          "143",
          "145",
          "146",
          "153",
          "148",
          "149",
          "151",
          "152",
          "165",
          "167",
          "169",
          "170",
          "171",
          "172",
          "176",
          "177",
          "178",
          "179",
          "181",
          "182",
          "186",
          "202",
          "204",
          "205",
          "208",
          "211",
          "213",
          "214",
          "216",
          "217",
          "218",
          "224",
          "226",
          "230",
          "231",
          "232",
          "233",
          "235",
          "236",
          "239",
          "240",
          "320",
          "241",
          "322",
          "244",
          "245",
          "249",
          "250",
          "256",
          "257",
          "260",
          "261",
          "263",
          "265",
          "271",
          "272",
          "273",
          "275",
          "276",
          "280",
          "284",
          "285",
          "286",
          "287",
          "288",
          "290",
          "291",
          "295",
          "297",
          "300",
          "301",
          "304",
          "305",
          "312",
          "316",
          "317",
          "318",
          "398",
          "399",
          "400",
          "407",
          "402",
          "403",
          "405",
          "408",
          "410",
          "416",
          "417",
          "418",
          "420",
          "421",
          "422",
          "423",
          "425",
          "426",
          "444",
          "430",
          "433",
          "434",
          "435",
          "436",
          "438",
          "440",
          "441",
          "475",
          "478",
          "480",
          "481",
          "483",
          "484",
          "489",
          "491",
          "494",
          "495",
          "498",
          "499",
          "500",
          "503",
          "504",
          "505",
          "508",
          "507",
          "529",
          "531",
          "535",
          "536",
          "538",
          "541",
          "543",
          "545",
          "560",
          "561",
          "562",
          "563",
          "564",
          "309",
          "576",
          "577",
          "578",
          "445",
          "580",
          "581",
          "599",
          "583",
          "854",
          "584",
          "588",
          "592",
          "593",
          "595",
          "598",
          "601",
          "604",
          "607",
          "608",
          "609",
          "611",
          "638",
          "614",
          "615",
          "616",
          "619",
          "620",
          "623",
          "624",
          "625",
          "626",
          "630",
          "631",
          "635",
          "636",
          "678",
          "710",
          "680",
          "681",
          "683",
          "684",
          "686",
          "687",
          "689",
          "691",
          "694",
          "697",
          "698",
          "700",
          "702",
          "704",
          "707",
          "729",
          "732",
          "734",
          "736",
          "790",
          "738",
          "739",
          "740",
          "742",
          "743",
          "746",
          "747",
          "748",
          "791",
          "749",
          "751",
          "753",
          "755",
          "758",
          "759",
          "761",
          "762",
          "765",
          "766",
          "768",
          "771",
          "777",
          "778",
          "781",
          "783",
          "831",
          "832",
          "833",
          "834",
          "837",
          "844",
          "845",
          "846",
          "848",
          "849",
          "850",
          "851",
          "853",
          "857",
          "858",
          "859",
          "886",
          "887",
          "889",
          "890",
          "892",
          "893",
          "895",
          "785",
          "905",
          "908",
          "092",
          "915",
          "918",
          "921",
          "922",
          "924",
          "925",
          "927",
          "931",
          "934",
          "935",
          "936",
          "941",
          "946",
          "976",
          "977",
          "980",
          "981",
          "989",
          "992"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
  }
}
"""

mk_payload = """

{
  "query": [
    {
      "code": "Alue 2021",
      "selection": {
        "filter": "item",
        "values": [
          "SSS",
          "MK01",
          "MK02",
          "MK04",
          "MK05",
          "MK06",
          "MK07",
          "MK08",
          "MK09",
          "MK10",
          "MK11",
          "MK12",
          "MK13",
          "MK14",
          "MK15",
          "MK16",
          "MK17",
          "MK18",
          "MK19",
          "MK21"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
  }
}

"""

sk_payload = """

{
  "query": [
    {
      "code": "Alue 2021",
      "selection": {
        "filter": "item",
        "values": [
          "SSS",
          "SK011",
          "SK014",
          "SK015",
          "SK016",
          "SK021",
          "SK022",
          "SK023",
          "SK024",
          "SK025",
          "SK041",
          "SK043",
          "SK044",
          "SK051",
          "SK052",
          "SK053",
          "SK061",
          "SK063",
          "SK064",
          "SK068",
          "SK069",
          "SK071",
          "SK081",
          "SK082",
          "SK091",
          "SK093",
          "SK101",
          "SK103",
          "SK105",
          "SK111",
          "SK112",
          "SK113",
          "SK114",
          "SK115",
          "SK122",
          "SK124",
          "SK125",
          "SK131",
          "SK132",
          "SK133",
          "SK134",
          "SK135",
          "SK138",
          "SK141",
          "SK142",
          "SK144",
          "SK146",
          "SK152",
          "SK153",
          "SK154",
          "SK161",
          "SK162",
          "SK171",
          "SK173",
          "SK174",
          "SK175",
          "SK176",
          "SK177",
          "SK178",
          "SK181",
          "SK182",
          "SK191",
          "SK192",
          "SK193",
          "SK194",
          "SK196",
          "SK197",
          "SK211",
          "SK212",
          "SK213"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
  }
}

"""

area_selections = [{'label':'Maakunta', 'value': 'Maakunta'},
                   {'label':'Seutukunta', 'value': 'Seutukunta'},
                   {'label':'Kunta', 'value': 'Kunta'}]

feature_selections = [{'label':feature, 'value':feature} for feature in sorted(requests.get(url).json()['variables'][1]['valueTexts'])]

queries = {'Maakunta':mk_payload,
           'Seutukunta':sk_payload,
           'Kunta':kunta_payload}

# Alueiden nimiä on muunnettava, koska ne ovat erit kartta-aineistossa ja Avainluvut-rajapinnassa.

maakunta_perusmuoto = {'Lapin maakunta':'Lappi',
                       'Pohjois-Savon maakunta':'Pohjois-Savo',
                       'Pohjois-Karjalan maakunta':'Pohjois-Karjala',
                       'Keski-Suomen maakunta':'Keski-Suomi', 
                       'Pohjois-Pohjanmaan maakunta':'Pohjois-Pohjanmaa',
                       'Pirkanmaan maakunta':'Pirkanmaa', 
                       'Varsinais-Suomen maakunta':'Varsinais-Suomi', 
                       'Ahvenanmaa':'Ahvenanmaa',
                       'Pohjanmaan maakunta':'Pohjanmaa', 
                       'Etelä-Pohjanmaan maakunta':'Etelä-Pohjanmaa',
                       'Keski-Pohjanmaan maakunta':'Keski-Pohjanmaa', 
                       'Uudenmaan maakunta':'Uusimaa',
                       'Kanta-Hämeen maakunta':'Kanta-Häme', 
                       'Kymenlaakson maakunta':'Kymenlaakso',
                       'Etelä-Savon maakunta':'Etelä-Savo', 
                       'Etelä-Karjalan maakunta':'Etelä-Karjala',
                       'Päijät-Hämeen maakunta':'Päijät-Häme', 
                       'Satakunnan maakunta':'Satakunta', 
                       'Kainuun maakunta':'Kainuu'
                      }

seutukunta_perusmuoto = {'Etelä-Pirkanmaan seutukunta': 'Etelä-Pirkanmaa',
                         'Forssan seutukunta': 'Forssa',
                         'Haapavesi-Siikalatvan seutukunta': 'Haapavesi-Siikalatva',
                         'Helsingin seutukunta': 'Helsinki',
                         'Hämeenlinnan seutukunta': 'Hämeenlinna',
                         'Imatran seutukunta': 'Imatra',
                         'Itä-Lapin seutukunta': 'Itä-Lappi',
                         'Jakobstadsregionen': 'Jakobstadsregionen',
                         'Joensuun seutukunta': 'Joensuu',
                         'Joutsan seutukunta': 'Joutsa',
                         'Jyväskylän seutukunta': 'Jyväskylä',
                         'Jämsän seutukunta': 'Jämsä',
                         'Järviseudun seutukunta': 'Järviseutu',
                         'Kajaanin seutukunta': 'Kajaani',
                         'Kaustisen seutukunta': 'Kaustinen',
                         'Kehys-Kainuun seutukunta': 'Kehys-Kainuu',
                         'Kemi-Tornion seutukunta': 'Kemi-Tornio',
                         'Keski-Karjalan seutukunta': 'Keski-Karjala',
                         'Keuruun seutukunta': 'Keuruu',
                         'Koillis-Savon seutukunta': 'Koillis-Savo',
                         'Koillismaan seutukunta': 'Koillismaa',
                         'Kokkolan seutukunta': 'Kokkola',
                         'Kotka-Haminan seutukunta': 'Kotka-Hamina',
                         'Kouvolan seutukunta': 'Kouvola',
                         'Kuopion seutukunta': 'Kuopio',
                         'Kuusiokuntien seutukunta': 'Kuusiokunnat',
                         'Lahden seutukunta': 'Lahti',
                         'Lappeenrannan seutukunta': 'Lappeenranta',
                         'Loimaan seutukunta': 'Loimaa',
                         'Lounais-Pirkanmaan seutukunta': 'Lounais-Pirkanmaa',
                         'Loviisan seutukunta': 'Loviisa',
                         'Luoteis-Pirkanmaan seutukunta': 'Luoteis-Pirkanmaa',
                         'Mariehamns stad': 'Mariehamns stad',
                         'Mikkelin seutukunta': 'Mikkeli',
                         'Nivala-Haapajärven seutukunta': 'Nivala-Haapajärvi',
                         'Oulun seutukunta': 'Oulu',
                         'Oulunkaaren seutukunta': 'Oulunkaari',
                         'Pieksämäen seutukunta': 'Pieksämäki',
                         'Pielisen Karjalan seutukunta': 'Pielisen Karjala',
                         'Pohjois-Lapin seutukunta': 'Pohjois-Lappi',
                         'Pohjois-Satakunnan seutukunta': 'Pohjois-Satakunta',
                         'Porin seutukunta': 'Pori',
                         'Porvoon seutukunta': 'Porvoo',
                         'Raahen seutukunta': 'Raahe',
                         'Raaseporin seutukunta': 'Raasepori',
                         'Rauman seutukunta': 'Rauma',
                         'Riihimäen seutukunta': 'Riihimäki',
                         'Rovaniemen seutukunta': 'Rovaniemi',
                         'Saarijärvi-Viitasaaren seutukunta': 'Saarijärvi-Viitasaari',
                         'Salon seutukunta': 'Salo',
                         'Savonlinnan seutukunta': 'Savonlinna',
                         'Seinäjoen seutukunta': 'Seinäjoki',
                         'Sisä-Savon seutukunta': 'Sisä-Savo',
                         'Suupohjan seutukunta': 'Suupohja',
                         'Sydösterbotten': 'Sydösterbotten',
                         'Tampereen seutukunta': 'Tampere',
                         'Torniolaakson seutukunta': 'Torniolaakso',
                         'Tunturi-Lapin seutukunta': 'Tunturi-Lappi',
                         'Turun seutukunta': 'Turku',
                         'Vaasan seutukunta': 'Vaasa',
                         'Vakka-Suomen seutukunta': 'Vakka-Suomi',
                         'Varkauden seutukunta': 'Varkaus',
                         'Ylivieskan seutukunta': 'Ylivieska',
                         'Ylä-Pirkanmaan seutukunta': 'Ylä-Pirkanmaa',
                         'Ylä-Savon seutukunta': 'Ylä-Savo',
                         'Äänekosken seutukunta': 'Äänekoski',
                         'Åboland-Turunmaan seutukunta': 'Åboland-Turunmaa',
                         'Ålands landsbygd': 'Ålands landsbygd',
                         'Ålands skärgård': 'Ålands skärgård'}

# https://geo.stat.fi/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=tilastointialueet:kunta1000k_2021&outputFormat=json
with open('kunnat_tk_4326.json', encoding = 'ISO-8859-1') as f:
    kuntarajat = json.load(f)

    
# https://geo.stat.fi/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=maakunta1000k_2021&outputFormat=json
with open('maakunnat_tk_4326.json', encoding = 'ISO-8859-1') as f:
    maakuntarajat = json.load(f)
    
# https://geo.stat.fi/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=tilastointialueet:seutukunta1000k_2021&outputFormat=json    
with open('seutukunnat_tk_4326.json', encoding = 'ISO-8859-1') as f:
    seutukuntarajat = json.load(f)

geojson_map = {'Kunta':kuntarajat, 'Maakunta':maakuntarajat, 'Seutukunta': seutukuntarajat} 

spinners = ['graph', 'cube', 'circle', 'dot' ,'default']

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = Dash(name = __name__, 
           #prevent_initial_callbacks = True, 
           server = server,
           meta_tags = [{'name':'viewport',
                        'content':'width=device-width, initial_scale=1.0, maximum_scale=1.2, minimum_scale=0.5'}],
           external_stylesheets=[dbc.themes.JOURNAL,"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"])


app.title = 'Suomen avainklusterit'



# Haetaan data aluetasoon perustuen.
def get_data(aluetaso):
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
 'Content-Type': 'application/json'}
    
    json = requests.post(url, data = queries[aluetaso], headers = headers).json()

    cities = list(json['dimension']['Alue 2021']['category']['label'].values())
    dimensions = list(json['dimension']['Tiedot']['category']['label'].values())
    values = json['value']

    cities_df = pd.DataFrame(cities, columns = ['Alue'])
    cities_df['index'] = 0
    dimensions_df = pd.DataFrame(dimensions, columns = ['dimensions'])
    dimensions_df['index'] = 0

    data = pd.merge(left = cities_df, right = dimensions_df, on = 'index', how = 'outer').drop('index',axis = 1)
    data['value'] = values
    data = pd.pivot_table(data, values = 'value', index = ['Alue'], columns = 'dimensions')
    koko_maa = data.loc['KOKO MAA']
    data = data.drop('KOKO MAA', axis=0)
    data['Aluejako'] = aluetaso
    if aluetaso == 'Maakunta':
        data.index = data.index.map(maakunta_perusmuoto)
    elif aluetaso == 'Seutukunta':
        data.index = data.index.map(seutukunta_perusmuoto)
    
    return (koko_maa, data)


# Haetaan alueiden kuntien määrät. 
# Näitä käytetään kahden muuttujan tarkastelussa sirontakuvion muotojen koon määrittelyssä.
# Lähde: https://www2.stat.fi/fi/luokitukset/alue/
def get_area_counts():
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
 'Content-Type': 'application/json'}
    
    kunta_seutu_url = 'https://data.stat.fi/api/classifications/v2/correspondenceTables/kunta_1_20210101%23seutukunta_1_20210101/maps?content=data&format=json&lang=fi&meta=min'

    seutu_kunta = pd.DataFrame([{'Seutukunta':c['targetItem']['classificationItemNames'][0]['name'], 'Kunta':c['sourceItem']['classificationItemNames'][0]['name']} for c in requests.get(kunta_seutu_url, headers = headers).json()])
        
    kunta_maa_url = 'https://data.stat.fi/api/classifications/v2/correspondenceTables/kunta_1_20210101%23maakunta_1_20210101/maps?content=data&format=json&lang=fi&meta=min'
    maa_kunta = pd.DataFrame([{'Maakunta':c['targetItem']['classificationItemNames'][0]['name'], 'Kunta':c['sourceItem']['classificationItemNames'][0]['name']} for c in requests.get(kunta_maa_url, headers = headers).json()])
    
    aluejako = pd.merge(left = seutu_kunta, right = maa_kunta, how = 'inner', on = 'Kunta')
    aluejako.Kunta = aluejako.Kunta.str.replace('Maarianhamina - Mariehamn','Maarianhamina')
    maakunnassa_kuntia = aluejako.groupby('Maakunta').Kunta.count()
    seutukunnassa_kuntia = aluejako.groupby('Seutukunta').Kunta.count()
    kunnassa_kuntia = aluejako.groupby('Kunta').Kunta.count()
    
    return {'Kunta':kunnassa_kuntia,
            'Seutukunta':seutukunnassa_kuntia,
            'Maakunta':maakunnassa_kuntia}

kuntamäärät_alueittain = get_area_counts()


def cluster_data(n_clusters, data, features):
    
    scl = StandardScaler()  

    x = data[features]

    X = scl.fit_transform(x)

    kmeans = KMeans(n_clusters, random_state = 42)
    
    preds = kmeans.fit_predict(X)

    clusters = preds + 1 
    
    inertia = round(kmeans.inertia_,1)
    silhouette = round(silhouette_score(x, preds),1)

    data['cluster'] = clusters
    data['inertia'] = inertia
    data['silhouette'] = silhouette
    data['features'] = '; '.join(features)
    data['pca'] = False
        
    color_df = colors.sample(len(colors)).copy()
    color_df.index = np.arange(len(colors))+1
    
    color_df = color_df.iloc[:len(pd.unique(data.cluster))]
    
    
    
    data = pd.merge(left=data.reset_index(),right=color_df, left_on='cluster', right_on=color_df.index).drop('Unnamed: 0',axis=1)
    
    
    
    data = data.sort_values(by='cluster')
    
    data.cluster = data.cluster.astype(str)
    
    data = data.set_index('Alue')
    

    return data

def test_clusters(data, features):
    
       
    return pd.DataFrame([{'clusters':k, 'inertia':(KMeans(n_clusters = k, random_state = 42).fit(StandardScaler().fit_transform(data[features]))).inertia_} for k in range(2,11)])

def test_clusters_with_PCA(data, features):
    
    X = StandardScaler().fit_transform(data[features])
    pca = PCA(n_components=len(features))
    principalComponents = pca.fit_transform(X)
    PCA_components = pd.DataFrame(principalComponents)

    return pd.DataFrame([{'clusters':k, 'inertia':KMeans(n_clusters = k, random_state = 42).fit(PCA_components.iloc[:,:3]).inertia_} for k in range(2,11)])

def cluster_data_with_PCA(n_clusters, data, features):
    
    scl = StandardScaler()  

    x = data[features]

    X = scl.fit_transform(x)
    
    pca = PCA(n_components = len(features))

    principalComponents = pca.fit_transform(X)
    
    PCA_components = pd.DataFrame(principalComponents)

    kmeans = KMeans(n_clusters, random_state = 42)
        
    preds = kmeans.fit_predict(PCA_components.iloc[:,:2])

    clusters = preds + 1 
    
    inertia = round(kmeans.inertia_,1)
    silhouette = round(silhouette_score(x, preds),1)

    data['cluster'] = clusters
    data['inertia'] = inertia
    data['silhouette'] = silhouette
    data['features'] = '; '.join(features)
    data['pca'] = True
        
    color_df = colors.sample(len(colors)).copy()
    color_df.index = np.arange(len(colors))+1
    
    color_df = color_df.iloc[:len(pd.unique(data.cluster))]
    
    
    
    data = pd.merge(left=data.reset_index(),right=color_df, left_on='cluster', right_on=color_df.index).drop('Unnamed: 0',axis=1)
    
    
    
    data = data.sort_values(by='cluster')
    
    data.cluster = data.cluster.astype(str)
    
    data = data.set_index('Alue')
    

    return data

def choose_value(primary, secondary, unit):
    
    if unit == '%' or '/' in unit:
        return primary
    else:
        return round(secondary,1)
def label_value(value, primary):
    
    if value == primary:
        return 'Koko maa'
    else:
        return 'Keskiarvo'    
    
def get_baseline(data, koko_maa):
    
    suomi = pd.DataFrame(koko_maa)
    suomi['yksikkö'] = [c.split(',')[-2].strip().split()[0] if len(c.split(',')[-2].strip().split())==1 or 'huoltosuhde' in c.split(',')[-2].strip().split() else '' for c in suomi.index]
    suomi.yksikkö = suomi.yksikkö.str.replace('Taloudellinen', '%')
    suomi.yksikkö = suomi.yksikkö.str.replace('Työpaikkaomavaraisuus', 'tpo')
    suomi.yksikkö = suomi.yksikkö.str.replace('Väkiluku', 'henkilöä')
    
    suomi['ka'] = data.drop(['Aluejako','inertia', 'silhouette', 'features', 'cluster','color'],axis=1).mean()
    suomi['arvo'] = suomi.apply(lambda row: choose_value(row['KOKO MAA'], row['ka'], row['yksikkö']),axis = 1)

    suomi['label'] = suomi.apply(lambda row: label_value(row['KOKO MAA'], row['arvo']),axis = 1)
    suomi.drop(['KOKO MAA','ka'],axis=1, inplace = True)
    suomi.yksikkö = suomi.yksikkö.str.replace('tpo','%')
    
    return suomi



def plot_feature(suomi, data, koko_maa, single_feature):
    
    aluejako = data.Aluejako.values[0]
    kuntamäärät = kuntamäärät_alueittain[aluejako]
    
    value_counts = pd.merge(left = data, right = kuntamäärät, left_on = data.index, right_on = kuntamäärät.index, how = 'inner').groupby('cluster').Kunta.sum()
    
    
    grouped_data = data.groupby(['cluster','color']).mean().reset_index().set_index('cluster')
    grouped_data.index = grouped_data.index.astype(int)
    grouped_data= grouped_data.sort_index()
    
    aluejako = data.Aluejako.values[0].lower()
    
    koko_maa_values = [suomi.loc[single_feature].arvo for i in range(len(grouped_data))]
    koko_maa_ka = koko_maa_values[0]
    
    
    unit = suomi.loc[single_feature].yksikkö
    label = suomi.loc[single_feature].label
    
    n_clusters = data['cluster'].value_counts().sort_index()
    n_clusters.index = n_clusters.index.astype(str)
    
    gd = grouped_data[[single_feature]].reset_index()
    gd = gd.pivot(columns='cluster').fillna(method='ffill').fillna(method='bfill').drop_duplicates()
    clusters_dict ={c[1]:c[1] for c in gd.columns}
    gd.columns = clusters_dict.keys()

    cluster_colors = grouped_data.loc[clusters_dict.keys()]['color'].drop_duplicates()

    
    traces = [go.Bar(x = [str(clusters_dict[i])], 
                         y = np.round(gd[i].values,1), 
                         name = clusters_dict[i], 
                         legendrank=int(i),
                         textposition = 'auto',
                     hovertemplate = ('<b>Klusteri </b><b>'+str(i)+'</b>:'+'<br>'+single_feature+': '+'{:,}'.format(np.round(gd[i].values[0],1)).replace('.0','').replace(',',' ')+''+unit+'<br>Klusterin koko: '+str(n_clusters.loc[str(i)])+' '+aluejako.lower().replace('ta','taa')+', '+str(value_counts[str(i)])+' kuntaa').replace(str(value_counts[str(i)])+' kuntaa'+', '+str(value_counts[str(i)])+' kuntaa',str(value_counts[str(i)])+' kuntaa').replace(' 1 kuntaa',' yksi kunta').replace(' 1 maakuntaa',' yksi maakunta').replace(' 1 seutukuntaa',' yksi seutukunta'),
                         textfont = dict(family='Arial Black', size = 18),
                         text = '{:,}'.format(np.round(gd.T.loc[i],1).values[0]).replace('.0','').replace(',',' ')+' '+unit+'<br>(N = '+str(n_clusters.loc[str(i)])+')</br>',
                         marker = dict(color = cluster_colors.loc[i])) for i in sorted(clusters_dict.keys())]

    
#     traces.append(go.Scatter(x = grouped_data.index.astype(str), 
#                                       y = koko_maa_values, 
#                                       name = '{} (~{} {})'.format(label, '{:,}'.format(round(koko_maa_ka,1)).replace(',',' '),unit),
#                                      mode = 'lines',
#                                       marker = dict(color='black'),
#                                       hoverinfo='none',
#                                       line = dict(width =6, dash ='dot')))
    
    
    figure = go.Figure(data = traces,

                   layout = go.Layout(title = dict(text = '<b>'+single_feature+'</b>'+'<br>Keskiarvot klustereittain sekä '+(label.lower()+'n vertailuarvo').replace('keskiarvon vertailuarvo', 'koko maan keskiarvo')+' (N = '+aluejako.replace('ta','tien')+' määrä klusterissa)'+'</br>', x=.5,font=dict(family='Arial',size=24)),
                                      legend = dict(title = '<b>Klusterit</b>', font=dict(size=18)),
                                      hoverlabel = dict(font_size = 16, font_family = 'Arial'),

                                     # width = 1000,
                                      height = 800,
                                    #  
                                     xaxis = dict(title=dict(text = 'Klusterit',font=dict(size=20, family = 'Arial Black')), 
                                                  tickfont = dict(family = 'Arial Black', size = 16)),
                                     yaxis = dict(title=dict(text = 'Klusterin keskiarvo ({})'.format(unit).replace('()',''),
                                                            font=dict(size=20, family = 'Arial Black')),
                                                 tickformat = ' ',
                                                 tickfont = dict(family = 'Arial', size = 16))))
    
    figure.add_hline(y = suomi.loc[single_feature].arvo,
                    annotation_text =  '{} (~{} {})'.format(label, '{:,}'.format(round(koko_maa_ka,1)).replace(',',' '),unit),
                    annotation_position="top left",
                     annotation_font_size=20,
                     annotation_font_color="black",
                    line = dict(width =6, dash ='dot'))
    
    return figure


def plot_counts(data):
    
    
    
    cdf = data.groupby(['cluster','color']).count().iloc[:,0].reset_index()
    cdf.columns = ['cluster','color','n_clusters']
    cdf = cdf.set_index('cluster')
    cdf.index = cdf.index.astype(int)
    cdf = cdf.sort_index()
    cdf.index = cdf.index.astype(str)
    
    
    aluejako = data.Aluejako.values[0]
    kuntamäärät = kuntamäärät_alueittain[aluejako]
    
    value_counts = pd.merge(left = data, right = kuntamäärät, left_on = data.index, right_on = kuntamäärät.index, how = 'inner').groupby('cluster').Kunta.sum()



    traces = [go.Bar(y = [c], 
                     x = [cdf.loc[c].n_clusters], 
                     orientation = 'h', 
                     legendrank = int(c), 
                     name = c,
                     textposition = 'auto',
                     textfont_size=16,
                     hovertemplate = ('<b>Klusteri </b><b>'+c+'</b>:'+'<br>Klusterin koko: '+str(cdf.loc[c].n_clusters)+' '+aluejako.lower().replace('ta','taa')+', '+str(value_counts[c])+' kuntaa').replace(str(value_counts[c])+' kuntaa'+', '+str(value_counts[c])+' kuntaa',str(value_counts[c])+' kuntaa').replace(' 1 kuntaa',' yksi kunta').replace(' 1 maakuntaa',' yksi maakunta').replace(' 1 seutukuntaa',' yksi seutukunta'),
                     text = 'N = '+str(cdf.loc[c].n_clusters),
                     marker = dict(color=cdf.loc[c].color)) for c in cdf.index]

    figure = go.Figure(data = traces,
                      layout = go.Layout(title=dict(text='<b>Klusterit koottain</b><br>(N = '+aluejako.lower().replace('ta','tien')+' määrä klusterissa)',
                                                    x=.5,
                                                    
                                                    font=dict(family='Arial', size=30)),
                                        legend = dict(title = '<b>Klusterit</b>', font=dict(size=18)),
                                       #  width = 1000,
                                      height = 800,
                                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                       yaxis = dict(title=dict(text='Klusterit',font=dict(size=20, family = 'Arial Black')), 
                                                    
                                                  tickfont = dict(family = 'Arial Black', size =16)),
                                        xaxis=dict(title= dict(text='Klusterin koko',font=dict(size=20, family = 'Arial Black')), 
                                                   tickformat=' ',
                                                   tickfont = dict(family = 'Arial', size =16)) 
                                        )
                      )
    
    
    return figure


def plot_map(data, geojson, aluetaso):
    
    data.cluster = data.cluster.astype(int)
    data = data.sort_values(by='cluster')
    data.cluster = data.cluster.astype(str)
    color_discrete_map = data[['cluster','color']].drop_duplicates().set_index('cluster').to_dict()['color']
    
    

    fig = px.choropleth_mapbox(data[['cluster','color']].reset_index(), 
                               geojson=geojson, 
                               locations='Alue', 
                               color='cluster',
                               color_discrete_map = color_discrete_map,
                               mapbox_style="open-street-map",
                               featureidkey='properties.nimi',
                               zoom=3.7, 
                               center = {"lat": 64.961093, "lon": 27.590605},
                               
                               labels={'Alue':aluetaso, 'cluster':'Klusteri'}
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                      height=600,
                      hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                      legend = dict(title = '<b>Klusterit</b>', font=dict(size=18)),
                      )
    
    
    return fig


def plot_correlations(data, koko_maa, suomi, feature1, feature2):
    
    
    cdf = data.groupby(['cluster','color']).count().iloc[:,0].reset_index()
    cdf.columns = ['cluster','color','n_clusters']
    cdf = cdf.set_index('cluster')

    d = data[['cluster',feature1,feature2]].sort_values(by='cluster').groupby('cluster').mean()
    d.index = d.index.astype(int)
    d = d.sort_index()
    d.index = d.index.astype(str)
    
    aluejako = data.Aluejako.values[0]
    kuntamäärät = kuntamäärät_alueittain[aluejako]
    
    value_counts = pd.merge(left = data, right = kuntamäärät, left_on = data.index, right_on = kuntamäärät.index, how = 'inner').groupby('cluster').Kunta.sum()

    
    max_size = 60
    
    traces = [go.Scatter(x = np.array([round(d.loc[c][feature1],1)]), 
                     y = np.array([round(d.loc[c][feature2],1)]), 
                     text ='Klusteri '+c, 
                     textfont=dict(family="arial black",size=16,color="black"),
                     textposition='top center',
                     name = c, 
                     mode = 'markers+text', 
                     hovertemplate = ('<b>Klusteri </b><b>'+c+'</b>:'+'<br>'+feature1+': {:,}'.format(round(d.loc[c][feature1],1)).replace('.0','').replace(',',' ')+' '+suomi.loc[feature1].yksikkö+'<br>'+feature2+': {:,}'.format(round(d.loc[c][feature2],1)).replace('.0','').replace(',',' ')+' '+suomi.loc[feature2].yksikkö+'<br>Klusterin koko: '+(str(cdf.loc[c].n_clusters)+' '+aluejako.lower().replace('ta','taa')+', '+str(value_counts[c])+' kuntaa')).replace(str(value_counts[c])+' kuntaa'+', '+str(value_counts[c])+' kuntaa',str(value_counts[c])+' kuntaa').replace(' 1 kuntaa',' yksi kunta').replace(' 1 maakuntaa',' yksi maakunta').replace(' 1 seutukuntaa',' yksi seutukunta'),
                     marker_size = value_counts[c], 
                     marker = dict(sizemode='area',opacity=.8, sizeref=2*value_counts.max()/ max_size**2, line_width=2, color = data[data.cluster==c].color.values[0])) for c in d.index]
    
    
    traces.append(go.Scatter(x = np.array([suomi.loc[feature1].arvo]), 
                         y = np.array([suomi.loc[feature2].arvo]), 
                         name = 'Koko maan vertailukohta', 
                         text='Koko maan<br>vertailukohta</br>',
                         textfont=dict(family="arial black",size=16,color="black"), 
                         mode = 'markers+text',                          
                         hovertemplate = '<b>Koko maan vertailukohta</b>: <br>'+feature1+' ({})'.format(suomi.loc[feature1].label.lower()).replace(' (koko maa)','')+': {:,}'.format(round(suomi.loc[feature1].arvo,1)).replace('.0','').replace(',',' ')+'<br>'+feature2+' ({})'.format(suomi.loc[feature2].label.lower()).replace(' (koko maa)','')+': {:,}'.format(round(suomi.loc[feature2].arvo,1)).replace('.0','').replace(',',' '),                            
                         marker = dict(size=max_size,color='black',opacity=.2,sizemode='area',sizeref=2*value_counts.max()/ max_size**2,line_width=2), 
                         marker_symbol='pentagon-dot'))

       
        
    title_text = {True: '<b>'+feature1+'</b><br>vs.</br><b>'+feature2+'</b><br></br>',
                 False: '<b>'+feature1+'</b> vs. <b>'+feature2+'</b><br></br>'}[len(feature1) + len(feature2) > 100]    
        
    fig = go.Figure(data=traces, 
                layout=go.Layout(#width=1000,
                                 height=1000,
                                 hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                 xaxis=dict(title = dict(text=feature1, font=dict(size=18, family = 'Arial Black')), tickformat = ' ', tickfont = dict(size=14)), 
                                 yaxis=dict(title = dict(text=feature2, font=dict(size=18, family = 'Arial Black')), tickformat = ' ', tickfont = dict(size=14)),
                                 title = dict(text = title_text, x=.5, font=dict(size=20,family = 'Arial')),
                                 legend = dict(title = '<b>Klusterit</b>',font=dict(size=18))
                                            
                                )
               )
    
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    
    return fig




kunnat_koko_maa, kunnat_data = get_data('Kunta')
maakunnat_koko_maa, maakunnat_data = get_data('Maakunta')
seutukunnat_koko_maa, seutukunnat_data = get_data('Seutukunta')

koko_maa_dict = {'Kunta':kunnat_koko_maa,
                'Seutukunta':seutukunnat_koko_maa,
                'Maakunta':maakunnat_koko_maa}
data_dict = {'Kunta':kunnat_data,
                'Seutukunta':seutukunnat_data,
                'Maakunta':maakunnat_data}

max_clusters = {'Kunta':len(kunnat_data),
               'Maakunta':len(maakunnat_data),
               'Seutukunta':len(seutukunnat_data)}

initial_features = [f['label'] for f in feature_selections if '2018' in f['label']]


initial_n_clusters = 4


def serve_layout():
    
    return html.Div(children = [
        
              
               html.H1('Suomen avainklusterit',style={'textAlign':'center'}),
               html.Br(),
              
               html.H3('klusterointityökalu',style={'textAlign':'center'}),
                html.Br(),
              
                dbc.Row(children = [
                    
                    dbc.Col(children=[
                       html.H3('Valitse klusterointimuuttujat.',style={'textAlign':'center'}),
                       html.Br(),
                       dcc.Dropdown(id = 'features', 
                            options = feature_selections, 
                            value = initial_features,
                            multi = True,
                            placeholder = 'Valitse klusterointimuuttujat.'),

                       html.Br(),
                       dash_daq.BooleanSwitch(id = 'select_all', 
                                              label = dict(label = 'Valitse kaikki',style = {'font-size':20, 'fontFamily':'Arial Black'}), 
                                              on = False, 
                                              color = 'blue') 

                    ],xs =10, sm=8, md=5, lg=6, xl=6),
                
                  
                   
                   dbc.Col(children=[
                       html.H3('Valitse aluetaso',style={'textAlign':'center'}),
                       html.Br(),

                       html.Div([dbc.RadioItems(id = 'area', 
                                      options = area_selections,
                                      className="btn-group",
                                      inputClassName="btn-check",
                                      labelClassName="btn btn-outline-primary",
                                      labelCheckedClassName="active",
                                 
                                      value = 'Kunta',
                                     labelStyle={'font-size':22}
                                     )
                                ],
                                style = {'textAlign':'center'}
                               ),
                       html.Br(),
                       html.H3('Valitse klustereiden määrä.',style={'textAlign':'center'}),
                       dcc.Slider(id = 'n_clusters',
                                 min = 2,
                                 max = 10,
                                 value = initial_n_clusters,
                                 step = 1,
                                 marks = {2: {'label':'2', 'style':{'font-size':20, 'fontFamily':'Arial Black'}},
                                          5:{'label':'5', 'style':{'font-size':20, 'fontFamily':'Arial Black'}},
                                          7:{'label':'7', 'style':{'font-size':20, 'fontFamily':'Arial Black'}},
                                          10:{'label':'10', 'style':{'font-size':20, 'fontFamily':'Arial Black'}}
                                         # 12:{'label':'12', 'style':{'font-size':20, 'fontFamily':'Arial Black'}},
                                         # 15:{'label':'15', 'style':{'font-size':20, 'fontFamily':'Arial Black'}}
                                          }
                                 ),
                       html.Br(),
                       html.Div(id = 'slider_update', children = [html.P('Valitsit {} klusteria.'.format(initial_n_clusters),style = {'textAlign':'center', 'fontSize':18, 'fontFamily':'Arial Black'})]),
                       html.Br(),
                   ],xs =10, sm=8, md=5, lg=6, xl=6)
                ]),
        
        # 4. rivi
        dbc.Row([dash_daq.BooleanSwitch(id = 'pca_switch', 
                                         label = dict(label = 'Käytä pääkomponenttianalyysia',style = {'font-size':20, 'fontFamily':'Arial Black'}), 
                                          on = False, 
                                          color = 'blue')],
               justify='center',align="start"),
        html.Div([html.Br()]),
        dbc.Row([
                       dbc.Button('Klusteroi',
                                  id='cluster_button',
                                  n_clicks=0,
                                  outline=False,
                                  className="me-1",
                                  size='lg',
                                  color='warning',
                                  style = dict(fontSize=28)
                                  ),

                    html.Br()
                    
                     
                    ],justify='center'),
        
        html.Br(),
        
        dbc.Tabs(children= [
            
            dbc.Tab(label = 'Klusterit ja sijainnit',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28}, 
                    children = [

                
                html.Br(),
                dbc.Row(id = 'other_buttons',justify='end'),
                html.Br(),
                dbc.Row( id = 'count_and_map')
                
            ]),
            
            dbc.Tab(label = 'Avainluvut klustereittain',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    children = [

                        
                        html.Br(),
                        dbc.Row(id = 'cluster_and_extra_feature')
                
            ]),
            dbc.Tab(label = 'Klustereiden tarkastelu kahden avainluvun mukaan',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    children = [

                       
                        html.Br(),
                        html.Br(),
                        dbc.Row(id = 'correlations')
                
            ]),
            dbc.Tab(label = 'Klustereiden määrän arviointi',
                   tabClassName="flex-grow-1 text-center",
                   tab_style = {'font-size':28},
                   children = [
                   
                       html.Br(),
                       dbc.Row(id = 'cluster_analysis', justify = 'center')
                   
                   ]),
            dbc.Tab(label = 'Ohje ja esittely',
                   tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                   children = [
                       
                      dbc.Row(justify='center', children=[
                          
                          dbc.Col(xs =10, sm=8, md=5, lg=6, xl=6, children =[
                       
                               html.Br(),
                               html.P('Avainluvuista avainklustereihin', style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.Br(),
                               html.H4('Johdanto',style={'textAlign':'center'}),
                               html.Br(),
                               html.P('Tässä sovelluksessa voi jakaa Suomen kunnat, seutukunnat tai maakunnat avainklustereihin itse valittujen kuntien avainlukujen mukaan. Kuntien avainluvut ovat Tilastokeskuksen ylläpitämä data-aineisto, joka sisältää alueita koskevia tunnuslukuja. Tämä sovellus pyrkiikin täydentämään Kuntien avainluvut -palvelua mahdollistamalla kuntien, seutukuntien tai maakuntien ryhmittelyn käyttäjän tarpeen mukaisilla indikaattoreilla. Käyttäjä voi valita avainlukujen ja aluetason lisäksi myös haluttujen klustereiden määrän. Klustereiden määrän valintaan ei ole oikeaa tai väärää vastausta. Haluttua määrää lieneekin hyvä tarkastella käyttäjän substanssin kautta. Jos on esimerkiksi tarkoitus perustaa tietty määrä aluekehitystyöryhmiä, on mahdollista valita klustereita tuo samainen määrä. Käyttäjä voi myös kokeilla eri lähtöarvoja klustereiden muodostamiseksi. Klusterointi perustuu tässä sovelluksessa ohjaamattomaan koneoppimiseen perustuvaan K-Means -klusterointiin, missä K on valittujen klustereiden määrä. Sivun alalaidasta löytyy linkki Wikipedia-artikkeliin K-Means -klusteroinnista.',style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.Br(),
                               html.H4('Ohje',style={'textAlign':'center'}),
                               html.Br(),
                               html.P('1. Valitse valikosta halutut avainluvut klusterointimuuttujiksi. Voit myös valita kaikki valikon alla olevasta painikkeesta.', style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                               html.P('2. Valitse haluttu aluetaso aluepainikkeista.',style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                               html.P('3. Valitse klustereiden määrä vierittämällä valintapalkkia.',style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                               html.P('4. Valitse kytkimellä käytetäänkö pääkomponenttianalyysiä. Asiaa on selitetty alla olevalla tekstillä.',style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                               html.P('5. Klusteroi klikkaamalla "Klusteroi" -painiketta.',style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                               html.Br(),
                               html.H4('Pääkomponenttianalyysista',style={'textAlign':'center'}),
                               html.Br(),
                               html.P('Pääkomponenttianalyysi (englanniksi Principal Component Analysis, PCA) on dimension redusointitekniikka. Sen tavoitteena on löytää monidimensioisesta datasta ne komponentit, joiden avulla sen keskeisimmät piirteet voidaan esittää ilman, että merkittävää informaatiota menee hukkaan. Tässä sovelluksessa PCA paketoi käytetyt avainluvut kahteen pääkomponenttiin. Tämä on käytännöllinen toimenpide erityisesti silloin, kun klusterointia tehdään perustuen moneen avainlukuun. PCA:lla voi siten poistaa niitä muuttujia, jotka aiheuttavat datassa kohinaa. PCA:n haittapuoli on kuitenkin se, että pääkomponentteja ei pysty palauttamaan takaisin alkuperäisiin muuttujiin, jolloin menetetään tarkka tieto siitä mitä muuttujia klusteroinnissa lopulta hyödynnettiin.',style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.Br(),
                               html.H4('Klusterit ja sijainnit',style={'textAlign':'center'}),
                               html.Br(),
                               html.P('Klusterit ja sijainnit -välilehdellä käyttäjä voi tarkastella klustereiden kokoja (eli niiden sisältämien alueiden määrää) sekä niiden alueellista jakautumista. Tämä auttaa myös klustereiden määrän määrittelyssä, mikäli halutaan mahdollisimman tasaisesti jakautuneita klustereita. Pylväskuvion alle ilmestyy myös klusteroinnin inertia -ja siluettipisteet. Ne ovat indikaattoreita, jotka kuvaavat klustereiden jakautumista. Parhaassa tapauksessa klusterit sisältävät samanlaisia jäseniä, ja klusterit ovat kaukana toisistaan. Inertia kuvaa vain edellistä, kun taas siluetti kuvaa kokonaisuutta. Inertialle ei ole viitearvoa, mutta pienemmät arvot kertovat paremmasta klusterin sisäisestä jaosta. Siluetti saa arvoja -1 ja 1 väliltä. Teoriassa lähempänä ykköstä oleva siluettiarvo kuvaa hyvää klusterijakoa ja lähellä nollaa olevat arvot indikoivat samanlaisia klustereita. Käytännössä kuitenkin saavutettavat arvot riippuvat alkuperäisestä aineistosta, eikä ole viitearvoja siitä mikä on paras mahdollinen siluettiarvo, joka voidaan muodostaa ko. aineisto klusteroimalla. Tämänkin indikaattorin muutosta käyttäjä voi tarkastella klusterointiasetuksia muuttamalla.',style = {'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.P('Klikkaamalla "Lataa karttanäkymä" -painiketta, voi klustereiden maantieteellistä jakautumista tarkastella kartalla. Lisäksi klusteroinnin tulokset voi viedä Excel-tiedostoon klikkaamalla "Lataa tiedosto koneelle" -nappia. Tiedostoon tulostuu kuntien avainluvut klustereittain sekä klusteroinnin metatiedot (valitut muuttujat, klustereiden määrä, aluetasosekä laatuindikaattorit).',style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.Br(),
                               html.H4('Avainluvut klustereittain', style = {'text-align':'center'}),
                               html.Br(),
                               html.P('Avainluvut klustereittain -lehdellä voi tarkastella klustereiden eroja valittujen avainlukujen suhteen. Pylväskuvioiden avulla pystyy havainnoimaan avinlukujen klusterikohtaisia keskiarvoja sekä miten ne suhtautuvat koko maan viitearvoon. Koko maan arvo on suhteellisissa luvuissa (esim. työllisyysaste) Tilastokeskuksen ilmoittama viitearvo, ja määrällisissä luvuissa (esim. väkiluku) alueiden keskiarvo. Pylväitä voi tarkastella sekä klusteroinnissa käytettyjen avainlukujen että klusteroinnin ulkopuolisten avainlukujen mukaan.',style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.Br(),
                               html.H4('Klustereiden tarkastelu kahden avainluvun mukaan.',style={'textAlign':'center'}),
                               html.Br(),
                               html.P('Klustereita voi tarkastella myös kahden avainluvun mukaan sille varatulla välilehdellä. Näin pystyy tarkastelemaan avainlukujen välisiä korrelaatioita klustereittain sekä profiloimaan klustereita paremmin.',style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                               html.Br(),
                               html.Div(style={'text-align':'center'},children = [
                                   html.H4('Lähteet', style = {'text-align':'center'}),
                                   html.Br(),
                                   html.Label(['Tilastokeskus: ', 
                                            html.A('Kuntien avainluvut', href = "https://www.stat.fi/tup/alue/kuntienavainluvut.html#?year=2021&active1=SSS",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Tilastokeskus: ', 
                                            html.A('Paikkatietoaineistot', href = "https://www.tilastokeskus.fi/org/avoindata/paikkatietoaineistot.html",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Wikipedia: ', 
                                            html.A('K-Means -klusterointi (englanniksi)', href = "https://en.wikipedia.org/wiki/K-means_clustering",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Wikipedia: ', 
                                            html.A('Pääkomponenttianalyysi', href = "https://fi.wikipedia.org/wiki/P%C3%A4%C3%A4komponenttianalyysi",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Wikipedia: ', 
                                            html.A('Kyynärpäämetodi (englanniksi)', href = "https://en.wikipedia.org/wiki/Elbow_method_(clustering)",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Wikipedia: ', 
                                            html.A('käytetyt värit', href = "https://en.wikipedia.org/wiki/Lists_of_colors",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Codeacademy: ', 
                                            html.A('Inertia klusteroinnissa (englanniksi)', href = "https://www.codecademy.com/learn/machine-learning/modules/dspath-clustering/cheatsheet",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Label(['Towards Data Science: ', 
                                            html.A('Siluetti -pisteytyksen esittely (englanniksi)', href = "https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c",target="_blank")
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20})
                               ]),
                               html.Br(),
                               html.Br(),
                               html.H4('Tekijä', style = {'text-align':'center'}),
                               html.Br(),
                               html.Div(style = {'text-align':'center'},children = [
                                   html.I('Tuomas Poukkula', style = {'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.A('Seuraa LinkedIn:ssä', href='https://www.linkedin.com/in/tuomaspoukkula/', target = '_blank',style = {'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.A('tai Twitterissä.', href='https://twitter.com/TuomasPoukkula', target = '_blank',style = {'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                   html.Br(),
                                   html.Br(),
                                   html.Label(['Sovellus ', 
                                            html.A('GitHub:ssa', href='https://github.com/tuopouk/suomenavainklusterit')
                                           ],style={'textAlign':'center','font-family':'Arial', 'font-size':20})
                               ])
                          ])
                      ])

                           

                       
                       
      
                   
             ])
        ]),
        html.Br(),
        # Piilo-div, johon säilötään klusteridata ja ladattava xlsx-tiedosto.
        html.Div(id = 'hidden_data_div',
                  children= [
                            dcc.Store(id='data_store'),
                            dcc.Download(id = "download-component")
                           ]
                 )
  
               
              ])
        
  

@app.callback(
    Output('select_all','on'),
    [Input('features','value')]
)    
def update_switch(features):
            
    return len(features) == len(feature_selections)
        
    
@app.callback(

    Output('cluster_analysis','children'),
    [Input('features','value'),
    Input('area','value')]

)
def update_cluster_analysis(features, area):
    
    data = data_dict[area]
    
    test = test_clusters(data, features)
    test_with_pca = test_clusters_with_PCA(data, features)
    
#    "<b>%{text}</b><br><br>" +
#         "GDP per Capita: %{x:$,.0f}<br>" +
#         "Life Expectation: %{y:.0%}<br>"
    figure = go.Figure(data=[go.Scatter(x = test.clusters, 
                                        y  = np.round(test.inertia,1), 
                                        mode = 'lines+markers',
                                        line = dict(color='firebrick', width=4),
                                        hovertemplate = "<b>%{x} klusterilla</b>",
                                        name = 'K-Means'),
               go.Scatter(x = test_with_pca.clusters, 
                          y  = test_with_pca.inertia,
                          line = dict(color='royalblue', width=4),
                          mode = 'lines+markers',
                          name = 'K-Means ja PCA',
                          hovertemplate = "<b>%{x} klusterilla</b>"
                          )],
                      layout = go.Layout(height = 800, 
                                        title = dict(text = 'Klustereiden määrän arviointi '+area.lower()+'tasolla',
                                                      x=.5,
                                                      font=dict(family='Arial Black',size=28)
                                                      ),
                                        legend = dict(title = '<b>Klusterointitavat</b>', font=dict(size=18)),                                                      
                                        yaxis = dict(title=dict(text='Inertia',font=dict(size=20, family = 'Arial Black')), 
                                                    
                                                  tickfont = dict(family = 'Arial Black', size =16)),
                                        xaxis = dict(title= dict(text='Klustereiden määrä',font=dict(size=20, family = 'Arial Black')), 
                                                   tickformat=' ',
                                                   tickfont = dict(family = 'Arial', size =16)),
                                         hovermode="x unified"
                                         
                                        )
                      )
    
    return [dbc.Col(align = 'center',children =[
                    
                    html.H3('Klustereiden määrän arvoiminen'),
                    html.Br(),
                    html.P('Tarvittavaa klustereiden määrää voi substanssinäkökulman lisäksi arvioida myös datapohjaisesti. Optimaalisessa klusterijaossa klusterin alkiot ova lähellä toisiaan. Tätä suuretta voi hakea laskemalla inertia-suuret jokaisella mahdollisella määrällä klustereita ja tarkastelemalla niiden muutosta. Kun tämä esitetään oheisella viivakaaviolla eri määrillä klustereita, huomataan, että inertian laskee klustereiden määrän kasvaessa. Siinä kohtaa käyrää, jossa syntyy terävin kulma, on optimaalisin määrä klustereita. Tätä pistettä nimitetään myös kyynärpää-pisteeksi (elbow point), mikä perustuu käyrän muotoon.'),
                    html.Br(),
                    html.P('Testejä voi tehdä vain valitsemalla klusterointimuuttujat sekä aluetason. Oheisessa kuvaajassa näytetään inertiakäyrät klusteroinnilla sekä pääkomponenttianalyysiä hyödyntävällä klusteroinnilla.'),
                    html.Br(),
                    html.P('On myös mahdollista, että käyrä on hyvin pyöreä, jolloin selkeää kyynärpääpistettä ei löydy.')

                    ],
                    xs =12, sm=12, md=5, lg=3, xl=3),
            dbc.Col(align = 'center',children = [
                    dcc.Graph(id = 'inertia_curve', figure = figure)
                    ],
                    xs =12, sm=12, md=7, lg=9, xl=9)]

@app.callback(
    Output('other_buttons','children'),
    [Input('cluster_button','n_clicks')]
)
def update_buttons(n_clicks):
    
    if n_clicks > 0:
        
        return [dbc.Col(children=[

                dbc.Button(children=[html.I(className="fa fa-download mr-1"), 'Lataa tiedosto koneelle'],
                           id='download_button',
                           n_clicks=0,
                           outline=True,
                           size = 'lg',
                           color = 'primary'
                           ),
                dbc.Button('Lataa karttanäkymä',
                           id='map_button',
                           n_clicks=0,
                           outline=True,
                           size = 'lg',
                           color = 'success'
                           ),            
                
               ],xs =10, sm=8, md=5, lg=6, xl=7
                      )]


@app.callback(
    ServersideOutput('data_store','data'),
    [Input('cluster_button','n_clicks'),
     State('area', 'value'),
    State('n_clusters', 'value'),
    State('features','value'),
    State('pca_switch','on')]
)
def perform_clustering(n_clicks,area, n_clusters, features, pca):
    
    if n_clicks > 0:
        
        data = data_dict[area]
        
        
        data = {True: cluster_data_with_PCA(n_clusters, data, features),
                False: cluster_data(n_clusters, data, features)
               }[pca]
        #data = cluster_data(n_clusters, data, features)
        
        koko_maa = koko_maa_dict[area]
        
        suomi = get_baseline(data, koko_maa)
        
        return {'data':data.reset_index().to_dict('records'), 
                    'koko_maa':koko_maa.reset_index().to_dict('records'), 
                    'suomi':suomi.reset_index().to_dict('records') }
        
        


@app.callback(
    Output('count_and_map','children'),
    [Input('cluster_button','n_clicks'),
     State('data_store', 'data')]
)
def update_count_and_map(n_clicks, dataset):
    
    if n_clicks > 0:

        return [
                dbc.Col(
                        children=[
                            
                            dcc.Graph(id = 'count_plot'),
                                html.P('Tämä kuvaaja havainnollistaa kuinka paljon alueita on jokaisessa klusterissa.',
                                      style = {'font-size':18, 'font-family':'Arial'}),
                                html.P('Klusteroinnista on myös laskettu inertia, -ja siluettisuureet. Lisätietoa saa alla olevista linkeistä sekä "Ohje ja esittely" -välilehdellä.',
                                      style = {'font-size':18, 'font-family':'Arial'}),
                                html.Br(),
                                html.Div(id = 'metrics'),

                        ],xs =12, sm=12, md=6, lg=6, xl=6),
                 dbc.Col(
                         children=[
                          dcc.Loading(children=[html.Div(id = 'map_div')], type = spinners[random.randint(0,len(spinners)-1)])
                         ],xs =12, sm=12, md=6, lg=6, xl=6)
               ]
 

@app.callback(
    Output('correlations', 'children'),
    [Input('cluster_button','n_clicks'),
     State('data_store', 'data')]
)
def update_correlations(n_clicks, dataset):
    
        if n_clicks > 0:

            feature_names = sorted([f['label'] for f in feature_selections])
            
            return [
                    dbc.Col(
                            children=[
                                html.Br(),
                                html.Br(),
                                html.H3('Valitse ensimmäinen avainluku'),
                                dcc.Dropdown(id = 'feature1',
                                           
                                            options = [{'label':f, 'value':f} for f in feature_names], 
                                            value = feature_names[np.random.randint(len(feature_names))],
                                            multi = False,
                                            placeholder = 'Valitse ensimmäinen avainluku'
                                            ),
                                html.Br(),
                                html.H3('Valitse toinen avainluku'),
                                dcc.Dropdown(id = 'feature2',
                                            options = [{'label':f, 'value':f} for f in feature_names],
                                            value = feature_names[np.random.randint(len(feature_names))],
                                            multi = False,
                                            placeholder = 'Valitse toinen avainluku'
                                            ),
                                html.Br(),
                                html.Br(),
                                html.P('Viereisessä kuvaajassa esitetään kahden valitun muuttujan klusterikohtaiset keskiarvot sirontakuviona. Näin voidaan tarkastella muuttujien korrelaatiota sekä muodostaa kuva klustereiden profiileista.',
                                      style = {'font-size':20, 'font-family':'Arial'}),
                                html.P('Kuviossa on myös koko maan vertailukohta esitetty harmaalla viisikulmiolla.',
                                      style = {'font-size':20, 'font-family':'Arial'}
                                      ),
                                html.P('Viemällä hiiren valitun klusterin tai koko maan vertailukohdan päälle, voi nähdä alueen, jonka kyseinen piste peittää. Esimerkiksi viemällä hiiren koko maan vertailukohdan päälle, voi muodostuneesta alueesta havainnoida mitkä klusterit jäävät koko maan vertailuarvojen sisälle.',
                                       style = {'font-size':20, 'font-family':'Arial'}
                                      ),
                                html.P('Kuvaajan selitteessä olevista arvoista klikkaamalla voi valita mitä pisteitä kuvaajassa näytetään.',
                                      style = {'font-size':20, 'font-family':'Arial'}
                                      )
                                

                            ],xs =12, sm=12, md=12, lg=3, xl=3),
                     dbc.Col(id = 'correlation_div',xs =12, sm=12, md=12, lg=9, xl=9)
                             
                   ]

        
@app.callback(
    Output('correlation_div', 'children'),
    [Input('feature1', 'value'),
    Input('feature2', 'value'),
    Input('data_store','data')]
)
def update_correlation_plot(feature1, feature2, dataset):
    
    return dcc.Graph(id = 'correlation_plot', 
                     figure = plot_correlations(pd.DataFrame(dataset['data']).set_index('Alue'), 
                                          pd.DataFrame(dataset['koko_maa']).set_index('dimensions'), 
                                          pd.DataFrame(dataset['suomi']).set_index('dimensions'), 
                                          feature1, 
                                          feature2)
                     )
       
        
@app.callback(

    Output('feature2', 'options'),
    [Input('feature1', 'value')]

)
def update_feature2_dd(value):
    
    feature_names = sorted([f['label'] for f in feature_selections if f['label'] != value])
    return [{'label':f, 'value':f} for f in feature_names]


@app.callback(

    Output('feature1', 'options'),
    [Input('feature2', 'value')]

)
def update_feature1_dd(value):
    
    feature_names = sorted([f['label'] for f in feature_selections if f['label'] != value])
    return [{'label':f, 'value':f} for f in feature_names]
        

@app.callback(
    Output('cluster_and_extra_feature','children'),
    [Input('cluster_button','n_clicks'),
     State('data_store', 'data'),
    State('features','value')]
)
def update_cluster_and_extra_feature(n_clicks, dataset, cluster_features):
    
    if n_clicks > 0:
        
        extra_features = sorted([f['label'] for f in feature_selections if f['label'] not in cluster_features])
    
        return [dbc.Col(
                        children=[
                              html.H3('Tarkastele klustereita valitun klusterointikriteerin mukaan.',style={'textAlign':'center',
                                                                                                           'font-size':28}),
                              html.Br(),
                              html.P('Tässä voi tarkastella klustereita niiden muodostamisessa käytettyjen kriteerien mukaan',
                                    style = {'font-family':'Arial','font-size':20}),
                              html.P('Alla olevissa pylväissä esitetään valitun avainluvun keskiarvot klustereittain sekä koko maan vertailuarvo.',
                                     style = {'font-family':'Arial','font-size':20} ),
                              html.P('Kuvaajan selitteessä olevista arvoista klikkaamalla voi valita mitä pisteitä kuvaajassa näytetään.',
                                      style = {'font-size':20, 'font-family':'Arial'}
                                      ),
                              html.Br(),
                              dcc.Dropdown(id = 'single_feature', 
                                     placeholder = 'Valitse tarkasteltava avainluku',
                                     value = cluster_features[np.random.randint(len(cluster_features))],      
                                     options = [{'label':f,'value':f,'title':'Klusteroinnin avainluku'} for f in sorted(cluster_features)],
                                     multi = False),
                              html.Div(id = 'feature_graph_div')
                             

                        ],xs =12, sm=2, md=6, lg=6, xl=6),
                 dbc.Col(
                         children=[
                              html.H3('Tarkastele klustereita muun avainluvun mukaan.',style={'textAlign':'center',
                                                                                             'font-size':28}),
                              html.Br(),
                              html.P('Tässä voi tarkastella klustereita niiden muodostamisessa käyttämättömien kriteerien mukaan',
                                    style = {'font-family':'Arial','font-size':20}),
                              html.P('Alla olevissa pylväissä esitetään valitun avainluvun keskiarvot klustereittain sekä koko maan vertailuarvo.',
                                     style = {'font-family':'Arial','font-size':20} ),
                              html.P('Kuvaajan selitteessä olevista arvoista klikkaamalla voi valita mitä pisteitä kuvaajassa näytetään.',
                                      style = {'font-size':20, 'font-family':'Arial'}
                                      ),
                              html.Br(),
                              dcc.Dropdown(id = 'extra_feature', 
                                     placeholder = 'Valitse tarkasteltava avainluku', 
                                     value = extra_features[np.random.randint(len(extra_features))], 
                                     options =  [{'label':f,'value':f,'title':'Muu avainluku'} for f in extra_features],
                                          multi = False),
                              html.Div(id = 'extra_feature_graph_div')
                         ],xs =12, sm=12, md=6, lg=6, xl=6)
               ]
             




@app.callback(
    Output('feature_graph_div','children'),
    [Input('data_store','data'),
     Input('single_feature','value')]
)
def plot_feature_graph(dataset, single_feature):


    return dcc.Graph(id = 'feature_graph', figure = plot_feature(pd.DataFrame(dataset['suomi']).set_index('dimensions'),
                       pd.DataFrame(dataset['data']).set_index('Alue'),
                       pd.DataFrame(dataset['koko_maa']).set_index('dimensions'),
                       single_feature))

 
@app.callback(
    Output('extra_feature_graph_div','children'),
    [Input('data_store','data'),
     Input('extra_feature','value')]
)
def plot_extra_feature_graph(dataset,extra_feature):
    
    return dcc.Graph(id = 'extra_feature_graph', figure = plot_feature(pd.DataFrame(dataset['suomi']).set_index('dimensions'),
                       pd.DataFrame(dataset['data']).set_index('Alue'),
                       pd.DataFrame(dataset['koko_maa']).set_index('dimensions'),
                       extra_feature))


@app.callback(
    Output('count_plot', 'figure'),
    [Input('data_store','data')]
)
def plot_count_graph(dataset):
    
    return plot_counts(pd.DataFrame(dataset['data']).set_index('Alue'))




@app.callback(
    Output('metrics', 'children'),
   
    [Input('data_store','data')]
)
def update_metrics_label(dataset):
    
    
    data = pd.DataFrame(dataset['data']).set_index('Alue')
    inertia = data.inertia.values[0]
    silhouette = data.silhouette.values[0]
    
    return [html.A('Inertia', href = 'https://www.codecademy.com/learn/machine-learning/modules/dspath-clustering/cheatsheet', target="_blank",style = {'font-size':18, 'font-family':'Arial'}),
            ': ',
            inertia,
            ', ',
            html.A('Silhouette', href = 'https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c', target="_blank",style = {'font-size':18, 'font-family':'Arial'}),': ',silhouette]
    




@app.callback(
    Output('map_div','children'),
    [Input('map_button','n_clicks'),
     State('data_store','data')]
)
def plot_cluster_map(n_clicks, dataset):
    
    
    if n_clicks > 0:
        
        data = pd.DataFrame(dataset['data']).set_index('Alue')
        
        area = data.Aluejako.values[0]

        geojson = geojson_map[area]

        cluster_map = plot_map(data, geojson, area)
        
        

        return html.Div(children =[html.H3('Klusterit {}'.format(area.lower()).replace('kunta','kunnittain'), style ={'textAlign': 'center','fontSize':40, 'family':'Arial Black'}),html.Br(),
                                   dcc.Graph(id = 'cluster_map', figure = cluster_map),
                                   html.P('Jos kartta ei näy, kokeile toisella selaimella.',
                                          style={'font-size':20,'font-family':'Arial'}),
                                  html.P('Kartassa näkyy värikoodattuna mihin klusteriin kukin '+area.lower()+' kuuluu. Selitteestä voit kaksoisklikkaamalla valita yhden klusterin, johon kuuluvat alueet haluat näyttää kartalla. Voit myös yhdellä klikkauksella valita mitä klustereita näytetään. Karttaa voi liikuttaa hiiren vasemmalla napilla. Oikealla napilla pystyy kiertämään karttaa. Oikeasta yläkulmasta selitteen yläpuolelta löytyy valintatyökalut, joista voi myös tallentaa kartan kuvana.',
                                        {'font-size':20,'font-family':'Arial'}),

                                  ])
        

@app.callback(
    Output('slider_update','children'),
    [Input('n_clusters', 'value')]
)
def update_n_cluster_view(n_clusters):
    return html.P('Valitsit {} klusteria.'.format(n_clusters),style = {'textAlign':'center', 'fontSize':20, 'fontFamily':'Arial Black'})


@app.callback(
    Output("download-component", "data"),
    [Input("download_button", "n_clicks"),
    State('data_store','data')
    ]
    
)
def download(n_clicks, dataset):
    
    if n_clicks > 0:
        
        
        data = pd.DataFrame(dataset['data']).set_index('Alue')
        
        inertia = data.inertia.values[0]
        silhouette = data.silhouette.values[0]
        aluejako = data.Aluejako.values[0]
        features = data.features.values[0].split(';')
        features = '\n'.join([str(count+1)+': '+value.strip()+' ' for count, value in enumerate(features)])
        n_clusters = len(pd.unique(data.cluster))
        pca = data.pca.values[0]
        data.drop(['color','Aluejako','silhouette','inertia','features','pca'],axis=1,inplace=True)
        data.columns = data.columns.str.replace('cluster','Klusteri')
        
        metadata = pd.DataFrame([{'Aluejako':aluejako,
                                  'Klusterit':n_clusters,
                                  'Pääkomponenttianlyysi':{True:'Kyllä',False:'Ei'}[pca],
                                 'Klusteroinnin avainluvut':features,
                                 'Siluetti':silhouette,
                                 'Inertia':inertia}]).T.reset_index()
        metadata.columns = ['Tieto','Arvo']
        metadata = metadata.set_index('Tieto')
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        
        data.to_excel(writer, sheet_name= 'Data klusteroituna')#city+'_'+datetime.now().strftime('%d_%m_%Y'))

        metadata.to_excel(writer, sheet_name = 'Klusteroinnin metadata')
        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Klusteri_'+aluejako.lower().replace('ta','nittain')+'_'+str(n_clusters)+'_klusterilla_'+datetime.now().strftime('%d_%m_%Y')+'.xlsx')



@app.callback(
    Output('features','value'),
    [Input('select_all', 'on')]
)
def update_feature_list(on):
       

    if on:
        
        return [f['value'] for f in feature_selections]
    else:
        raise PreventUpdate
    


 
app.layout = serve_layout
if __name__ == "__main__":
    app.run_server(debug=False)