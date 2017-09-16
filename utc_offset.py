import datetime
import pytz
from tzwhere import tzwhere
def utc_offset(lat, lon):
    tzw = tzwhere.tzwhere()
    timezone_str = tzw.tzNameAt(lat, lon)
    timezone_str
    timezone = pytz.timezone(timezone_str)
    return timezone.utcoffset