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

# ---- Default initial directories for file browsers ----
DEFAULT_INPUTS_INITIALDIR = "./"
DEFAULT_SHPFILE_INITIALDIR = "./"

# Default online US counties shapefile (Census cartographic boundary 1:500k)
# Use 2020 by default to match legacy reports (pre-CT planning region change)
def _online_counties_url(year: str) -> str:
    y = str(year)
    return f"https://www2.census.gov/geo/tiger/GENZ{y}/shp/cb_{y}_us_county_500k.zip"

DEFAULT_ONLINE_COUNTIES_URL = _online_counties_url('2020')

# ---- GRIDDESC parsing toggle ----
# Toggle: set to True if you want spherical earth (a=b=6370000 m) instead of WGS84
USE_SPHERICAL_EARTH = True

def _config_file() -> str:
    """Return the path to the configuration file."""
    cfg_dir = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('./'), '.config')
    try:
        os.makedirs(cfg_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(cfg_dir, 'smkgui_settings.json')

def load_settings() -> dict:
    """Load the entire settings dictionary from the JSON config file."""
    try:
        cfg = _config_file()
        if not os.path.exists(cfg):
            return {}
        with open(cfg, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_settings(settings: dict) -> None:
    """Collect and save current GUI settings to the JSON config file."""
    try:
        cfg = _config_file()
        with open(cfg, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        # best-effort; ignore failures
        pass