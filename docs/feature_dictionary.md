# Traffic Feature Dictionary

| Feature | Type | Description |
|---|---|---|
| year | numerical | Calendar year |
| month | numerical | Month number |
| day | numerical | Day of month |
| hour | numerical | Hour of day |
| day_of_week | numerical | Day of week from 0 to 6 |
| week_of_year | numerical | ISO week number |
| day_of_year | numerical | Day number within year |
| is_weekend | binary | 1 if Saturday/Sunday |
| is_morning_peak | binary | 1 during morning peak |
| is_evening_peak | binary | 1 during evening peak |
| is_rush_hour | binary | 1 during defined rush hours |
| rain_flag | binary | 1 when rainfall > 0 |
| snow_flag | binary | 1 when snowfall > 0 |
| time_period | categorical | Night, morning, afternoon, evening, late night |
| temp | numerical | Temperature |
| rain_1h | numerical | Rainfall in previous hour |
| snow_1h | numerical | Snowfall in previous hour |
| clouds_all | numerical | Cloud coverage |
| holiday | categorical | Holiday indicator |
| weather_main | categorical | Main weather category |
| weather_description | categorical | Detailed weather condition |
| traffic_volume | target | Number of vehicles |