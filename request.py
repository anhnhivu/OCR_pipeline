import requests
from requests.exceptions import HTTPError
import json 

coors = [10.9508275005, 106.5872017546]
try:
    response = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?latlng={coors[0]},{coors[1]}&key=AIzaSyCaRkfQYahP3fTIL31Da9Ppv5rnNWcG1F0&language=vi')
     # access JSOn content
    jsonResponse = response.json()
    print(jsonResponse["results"][0]["formatted_address"])

except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')