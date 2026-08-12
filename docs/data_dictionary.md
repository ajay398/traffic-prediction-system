# Traffic Dataset Data Dictionary

| Column | Type | Description |
|---|---|---|
| holiday | object | Holiday information |
| temp | float | Temperature |
| rain_1h | float | Rainfall in the previous hour |
| snow_1h | float | Snowfall in the previous hour |
| clouds_all | integer | Cloud coverage |
| weather_main | object | Main weather category |
| weather_description | object | Detailed weather description |
| date_time | datetime | Date and time of observation |
| traffic_volume | integer | Traffic volume; prediction target |

## Target Variable

`traffic_volume`

The model will predict the number of vehicles passing through the monitored location during the corresponding time period.