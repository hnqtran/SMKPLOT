"""Configuration constants and settings for SMKPLOT GUI."""

import os
import json
from typing import Dict

# ---- Static mapping: State FIPS -> State name ----
US_STATE_FIPS_TO_NAME: Dict[str, str] = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia', '12': 'Florida',
    '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois', '18': 'Indiana',
    '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi',
    '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire',
    '34': 'New Jersey', '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota',
    '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', '49': 'Utah',
    '50': 'Vermont', '51': 'Virginia', '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin',
    '56': 'Wyoming',
    # Territories
    '60': 'American Samoa', '66': 'Guam', '69': 'Northern Mariana Islands', '72': 'Puerto Rico', '78': 'U.S. Virgin Islands'
}

# ---- Key cols to be kept in exported pivotted csv files
key_cols = ['country_cd','tribal_code','region_cd','facility_id','unit_id','rel_point_id','scc','latitude','longitude']

# Define constant column name lists for remapping (moved from data_processing.py)
COUNTRY_COLS = ['country_cd', 'country', 'country_id']
TRIBAL_COLS = ['tribal_code', 'tribal_name', 'tribe_id']
REGION_COLS = ['region_cd', 'regioncd', 'region', 'county_id', 'fips']
FACILITY_COLS = ['facility_id', 'facility']
UNIT_COLS = ['unit_id']
REL_COLS = ['rel_point_id', 'release_point_id', 'point_id']
EMIS_COLS = ['ann_value', 'emission']
SCC_COLS = ['scc']
POL_COLS = ['poll', 'pollutant']
LAT_COLS = ['latitude', 'lat']
LON_COLS = ['longitude', 'lon']

# Define country code mappings following ISO 3166-1 alpha-2 codes for Northern Hemisphere countries
COUNTRY_CODE_MAPPINGS = {
    'US': '0',    'CA': '1',    'MX': '2',    'RU': '3',    'CN': '4',
    'JP': '5',    'GB': '6',    'FR': '7',    'DE': '8',    'IT': '9',
    'ES': '10',   'KR': '11',   'TR': '12',   'IR': '13',   'SA': '14',
    'UA': '15',   'PL': '16',   'IQ': '17',   'AF': '18',   'PK': '19',
    'ID': '20',   'EG': '21',   'NG': '22',   'ET': '23',   'DZ': '24',
    'MA': '25',   'VE': '26',   'TH': '27',   'VN': '28',   'PH': '29',
    'SD': '30'
}

# ---- Default initial directories for file browsers ----
DEFAULT_INPUTS_INITIALDIR = "./"
DEFAULT_SHPFILE_INITIALDIR = "./"

# Default online US counties shapefile (Census cartographic boundary 1:500k)
# Use 2020 by default to match legacy reports (pre-CT planning region change)
DEFAULT_ONLINE_COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"

# ---- GRIDDESC parsing toggle ----
# Toggle: set to True if you want spherical earth (a=b=6370000 m) instead of WGS84
USE_SPHERICAL_EARTH = True