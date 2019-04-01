import csv
import requests
import pandas as pd

save_temp_dict = {}

df_of_codes = pd.read_excel('./Data/airport-codes.xls')
base_url = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"

def calculateaveragewind(data):
    total_wind = 0
    for day in data["data"]["weather"][0]["hourly"]:
        total_wind = total_wind + int(day["windspeedMiles"])
    
    average_wind = total_wind / len(data["data"]["weather"][0]["hourly"])
    return average_wind

def calculateaveragevisibility(data):
    total_visibility = 0
    for day in data["data"]["weather"][0]["hourly"]:
        total_visibility = total_visibility + int(day["visibility"])
    
    average_visibility = total_visibility / len(data["data"]["weather"][0]["hourly"])
    return average_visibility

with open('data/reduce_nashville_flights.csv', 'r') as read_csv_file, open('data/added_weather_fields.csv', mode='w', newline="") as write_csv_file:
    airline_reader = csv.reader(read_csv_file)
    airline_writer = csv.writer(write_csv_file)

    # add headers
    headers = next(airline_reader, None)

    headers.append("ORIGIN_MIN_TEMPERATURE")
    headers.append("ORIGIN_MAX_TEMPERATURE")
    headers.append("ORIGIN_SNOW_CM")
    headers.append("ORIGIN_AVG_WIND")
    headers.append("ORIGIN_AVG_VISIBILITY")

    headers.append("DESINATION_MIN_TEMPERATURE")
    headers.append("DESINATION_MAX_TEMPERATURE")
    headers.append("DESINATION_SNOW_CM")
    headers.append("DESINATION_AVG_WIND")
    headers.append("DESTINATION_AVG_VISIBILITY")

    airline_writer.writerow(headers)

    for row in airline_reader:
        added_row = row
        
        # departure city
        if row[1] + row[2] + row[7] in save_temp_dict:
            # add to row
            added_row.append(save_temp_dict[row[1] + row[2] + row[7] + "min"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[7] + "max"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[7] + "snow"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[7] + "wind"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[7] + "visibility"])
            airline_writer.writerow(row)
        else:
            airport_code = row[7]
            dataframe_airport_code = df_of_codes.loc[df_of_codes['Airport Code'] == airport_code]['City name']
            if(dataframe_airport_code.empty):
                print("airport not found")
                continue

            city_name_str = dataframe_airport_code.to_string().split('    ')[1]
            url = base_url + "?key=a4f3053f03ca44d890100412192903&q=" + city_name_str + "&date=2015-" + row[1] + "-" + row[2] + "&format=json"
            result = requests.get(url = url, params = {})
            data = result.json()

            min_temp = data["data"]["weather"][0]["mintempF"]
            max_temp = data["data"]["weather"][0]["maxtempF"]
            snow_cm = data["data"]["weather"][0]["totalSnow_cm"]
            max_wind = calculateaveragewind(data)
            visibility = calculateaveragevisibility(data)
            # add to dictionary
            save_temp_dict[row[1] + row[2] + row[7] + "min"] = min_temp
            save_temp_dict[row[1] + row[2] + row[7] + "max"] = max_temp
            save_temp_dict[row[1] + row[2] + row[7] + "snow"] = snow_cm
            save_temp_dict[row[1] + row[2] + row[7] + "wind"] = max_wind
            save_temp_dict[row[1] + row[2] + row[7] + "visibility"] = visibility
            
            # add to row
            added_row.append(min_temp)
            added_row.append(max_temp)
            added_row.append(snow_cm)
            added_row.append(max_wind)
            added_row.append(visibility)
        
        # arrival city
        if row[1] + row[2] + row[8] in save_temp_dict:
            # add to row
            added_row.append(save_temp_dict[row[1] + row[2] + row[8] + "min"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[8] + "max"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[8] + "snow"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[8] + "wind"])
            added_row.append(save_temp_dict[row[1] + row[2] + row[8] + "visibility"])
            airline_writer.writerow(row)
        else:
            airport_code = row[8]
            dataframe_airport_code = df_of_codes.loc[df_of_codes['Airport Code'] == airport_code]['City name']
            if(dataframe_airport_code.empty):
                print("airport not found")
                continue

            city_name_str = dataframe_airport_code.to_string().split('    ')[1]
            url = base_url + "?key=a4f3053f03ca44d890100412192903&q=" + city_name_str + "&date=2015-" + row[1] + "-" + row[2] + "&format=json"
            result = requests.get(url = url, params = {})
            data = result.json()

            min_temp = data["data"]["weather"][0]["mintempF"]
            max_temp = data["data"]["weather"][0]["maxtempF"]
            snow_cm = data["data"]["weather"][0]["totalSnow_cm"]
            max_wind = calculateaveragewind(data)
            visibility = calculateaveragevisibility(data)

            # add to dictionary
            save_temp_dict[row[1] + row[2] + row[8] + "min"] = min_temp
            save_temp_dict[row[1] + row[2] + row[8] + "max"] = max_temp
            save_temp_dict[row[1] + row[2] + row[8] + "snow"] = snow_cm
            save_temp_dict[row[1] + row[2] + row[8] + "wind"] = max_wind
            save_temp_dict[row[1] + row[2] + row[8] + "visibility"] = visibility

            # add to row
            added_row.append(min_temp)
            added_row.append(max_temp)
            added_row.append(snow_cm)
            added_row.append(max_wind)
            added_row.append(visibility)

            airline_writer.writerow(row)