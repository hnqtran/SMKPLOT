# Author: tranhuy@email.unc.edu
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
key_cols = ['country_cd','tribal_code','region_cd','facility_id','unit_id','rel_point_id','facility_name','scc','latitude','longitude']

# Constant column name lists for remapping
COUNTRY_COLS = ['country_cd', 'country', 'country_id']
TRIBAL_COLS = ['tribal_code', 'tribal_name', 'tribe_id']
REGION_COLS = ['region_cd', 'regioncd', 'region', 'county_id', 'fips', 'state_county_fips']
FACILITY_COLS = ['facility_id', 'facility', 'facility_name', 'oris']
UNIT_COLS = ['unit_id', 'unit_name']
REL_COLS = ['rel_point_id', 'release_point_id', 'point_id', 'stack_id']
EMIS_COLS = ['ann_value', 'emission', 'emissions', 'value', 'monthtot']
SCC_COLS = ['scc', 'scc_code', 'scc code', 'source_classification_code', 'source classification code']
DESC_COLS = ['scc description', 'scc_description', 'scc_desc', 'scc_description_name']
POL_COLS = ['poll', 'pollutant', 'poll_name', 'pollutant_name', 'poll_id']
LAT_COLS = ['latitude', 'lat', 'y_coord']
LON_COLS = ['longitude', 'lon', 'long', 'x_coord']

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

# ---- Tribal FIPS to County FIPS Mapping ----
# Used to map tribal area emissions (IDs starting with 88) to a primary underlying county
# for visualization on standard administrative maps.
# This covers major energy-producing and populated tribal lands across CONUS.
# 
# DEVELOPMENT METHODOLOGY:
# 1. Identified tribal area codes (88xxx) used in EPA emissions inventories (NEI/SMOKE).
# 2. Consulted EPA COSTCY/GE_DAT files (e.g., costcy_for_2017platform_20dec2023_v8.txt)
#    to identify the tribal name and state associated with each code.
# 3. Mapped the tribal land physical extents/major energy production centers to 
#    the primary underlying US county FIPS code.
# 4. Prioritized regions with significant oil & gas production (San Juan, Permian, 
#    Northern Plains, Oklahoma) to prevent emission "artifacts" (dropped data) 
#    during administrative attribute-based joins.
TRIBAL_TO_COUNTY_FIPS = {
    # -- San Juan / Permian / Four Corners --
    '88750': '08067',  # Southern Ute Reservation -> La Plata, CO
    '88755': '08083',  # Ute Mountain Reservation -> Montezuma, CO
    '88124': '35045',  # Navajo Nation -> San Juan, NM
    '88701': '35039',  # Jicarilla Apache Nation -> Rio Arriba, NM
    '88702': '35035',  # Mescalero Reservation -> Otero, NM
    '88703': '35006',  # Acoma -> Cibola, NM
    '88707': '35006',  # Laguna -> Cibola, NM
    '88714': '35039',  # San Juan / Ohkay Owingeh -> Rio Arriba, NM
    '88716': '35043',  # Santa Ana -> Sandoval, NM
    '88719': '35043',  # Zia -> Sandoval, NM
    '88720': '35031',  # Zuni -> McKinley, NM
    '44444': '35045',  # Generic San Juan Basin placeholder

    # -- Arizona / Nevada --
    '88603': '04012',  # Colorado River -> La Paz, AZ
    '88608': '04017',  # Hopi -> Navajo, AZ
    '88610': '04019',  # Tohono O'Odham -> Pima, AZ
    '88614': '04021',  # Gila River -> Pinal, AZ
    '88615': '04013',  # Salt River -> Maricopa, AZ
    '88616': '04007',  # San Carlos -> Gila, AZ
    '88617': '04015',  # Kaibab -> Mohave, AZ
    '88507': '04017',  # Fort Apache -> Navajo, AZ
    '88651': '32029',  # Pyramid Lake -> Washoe, NV

    # -- Oklahoma (Major Energy Producing Tribes) --
    '88151': '40113',  # Osage Nation -> Osage, OK
    '88118': '40013',  # Choctaw -> Bryan, OK
    '88043': '40021',  # Cherokee -> Cherokee, OK
    '88044': '40123',  # Chickasaw -> Pontotoc, OK
    '88114': '40111',  # Muscogee (Creek) -> Okmulgee, OK

    # -- Northern Plains / Mountain (Oil & Gas focuses) --
    '88770': '56013',  # Wind River -> Fremont, WY
    '88687': '49047',  # Uintah-Ouray -> Uintah, UT
    '88012': '30035',  # Blackfeet -> Glacier, MT
    '88049': '30003',  # Crow -> Big Horn, MT
    '88136': '30087',  # Northern Cheyenne -> Rosebud, MT
    '88077': '30085',  # Fort Peck -> Roosevelt, MT
    '88076': '30005',  # Fort Belknap -> Blaine, MT
    '88078': '38061',  # Fort Berthold -> Mountrail, ND
    '88301': '38005',  # Spirit Lake -> Benson, ND

    # -- South Dakota --
    '88344': '46031',  # Standing Rock -> Corson, SD
    '88029': '46041',  # Cheyenne River -> Dewey, SD
    '88235': '46102',  # Pine Ridge (Oglala Lakota) -> Oglala Lakota, SD
    '88283': '46121',  # Rosebud -> Todd, SD

    # -- Washington / Northwest --
    '88047': '53047',  # Colville -> Okanogan, WA
    '88315': '53077',  # Yakama -> Yakima, WA
    '88234': '16011',  # Fort Hall (Shoshone-Bannock) -> Bingham, ID

    # -- Great Lakes / Midwest --
    '88319': '55087',  # Oneida -> Outagamie, WI
    '88130': '27021',  # Leech Lake -> Cass, MN
    '88409': '27007',  # Red Lake -> Beltrami, MN
    '88410': '27005',  # White Earth -> Becker, MN

    # -- Southeast --
    '88122': '28099',  # Mississippi Band of Choctaw -> Neshoba, MS
    '88061': '37173',  # Eastern Band of Cherokee -> Swain, NC
}