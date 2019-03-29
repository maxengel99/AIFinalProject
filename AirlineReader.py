import csv
import requests

base_url = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"

save_temp_dict = {}

with open('data/flights.csv', 'r') as read_csv_file, open('data/nashville_flights.csv', mode='w', newline="") as write_csv_file:
    airline_reader = csv.reader(read_csv_file)
    airline_writer = csv.writer(write_csv_file)
    
    # https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
    # add additional headers
    headers = next(airline_reader, None)
    headers.append("MIN_TEMPERATURE")
    headers.append("MAX_TEMPERATURE")
    headers.append("SNOW_CM")
    airline_writer.writerow(headers)

    for row in airline_reader:

        if row[7] == 'BNA' or row[8] == 'BNA':
            added_row = row
            if row[1] + row[2] + "min" in save_temp_dict:
                print("using dict")

                # add to row
                added_row.append(save_temp_dict[row[1] + row[2] + "min"])
                added_row.append(save_temp_dict[row[1] + row[2] + "max"])
                added_row.append(save_temp_dict[row[1] + row[2] + "snow"])
                airline_writer.writerow(row)
            else:
                url = base_url + "?key=[add this later]&q=Nashville&date=2015-" + row[1] + "-" + row[2] + "&format=json"
                print("making request")
                result = requests.get(url = url, params = {})
                data = result.json()

                min_temp = data["data"]["weather"][0]["mintempF"]
                max_temp = data["data"]["weather"][0]["maxtempF"]
                snow_cm = data["data"]["weather"][0]["totalSnow_cm"]
                
                # add to dictionary
                save_temp_dict[row[1] + row[2] + "min"] = min_temp
                save_temp_dict[row[1] + row[2] + "max"] = max_temp
                save_temp_dict[row[1] + row[2] + "snow"] = snow_cm

                # add to row
                added_row.append(min_temp)
                added_row.append(max_temp)
                added_row.append(snow_cm)

                airline_writer.writerow(row)