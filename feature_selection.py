import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor

import data_analyzing

# Build a classification task using 3 informative features
X, y = data_analyzing.get_data()

selected = ["full_sq", "num_room", "life_sq", "kitch_sq", "floor", "timestamp", "state", "max_floor",
            "build_year", "material", "industrial_km", "cafe_count_5000_price_high", "cafe_count_2000",
            "public_transport_station_km", "sport_count_3000", "catering_km", "fitness_km", "kindergarten_km",
            "prom_part_1500", "trc_sqm_500", "office_sqm_5000", "green_part_500", "trc_count_3000",
            "product_type_OwnerOccupier"
            ]

maybe_dropped = [
    "cafe_count_3000_price_1500", "cafe_count_2000_price_2500",
    "cafe_count_3000_price_2500", "cafe_count_3000_price_1000", "cafe_count_5000_price_2500", "cafe_count_5000",
    "cafe_count_3000", "cafe_count_5000_price_1500",

    "prom_part_3000",
    "public_transport_station_min_walk",
]

dropped = ["build_count_before_1920", "build_count_1921-1945", "build_count_1946-1970", "build_count_1971-1995",
           "work_male", "build_count_panel", "0_6_female", "young_all", "raion_popul", "mosque_count_3000", "area_m",
           "0_13_all", "young_female", "preschool_quota", "cafe_count_500_price_high", "7_14_female",
           "cafe_count_1000_price_high", "raion_build_count_with_builddate_info",
           "raion_build_count_with_material_info", "work_female", "school_quota", "7_14_male", "0_6_male",
           "ekder_female", "16_29_all", "work_all", "mosque_count_2000", "ekder_male", "0_17_male", "0_6_all",
           "build_count_frame", "male_f", "mosque_count_1000", "female_f", "0_13_male", "ekder_all", "build_count_wood",
           "full_all", "mosque_count_1500", "children_preschool", "7_14_all", "16_29_male", "0_17_all", "0_13_female",
           "16_29_female", "young_male", "children_school", "0_17_female", "build_count_mix",
           "build_count_foam", "mosque_count_500", "water_1line", "culture_objects_top_25", "thermal_power_plant_raion",
           "big_road1_1line", "incineration_raion", "oil_chemistry_raion", "radiation_raion", "railroad_1line",
           "railroad_terminal_raion", "big_market_raion", "nuclear_reactor_raion", "detention_facility_raion",

           "cafe_count_1000_price_1500", "cafe_count_1000_price_1000", "museum_km", "cafe_count_500_price_500",
           "cafe_count_3000_na_price", "prom_part_1000", "railroad_station_walk_min", "exhibition_km",
           "church_count_2000", "cafe_count_1000_na_price", "cafe_sum_1500_max_price_avg", "church_count_1500",
           "office_count_2000", "cafe_avg_price_1500", "indust_part", "cafe_avg_price_1000", "incineration_km",
           "sadovoe_km", "cafe_sum_1000_min_price_avg", "trc_sqm_2000", "trc_count_500", "green_part_2000",
           "stadium_km", "trc_count_1500", "trc_sqm_5000", "cafe_sum_1000_max_price_avg", "cafe_count_1500_price_1000",
           "bus_terminal_avto_km", "hospital_beds_raion", "office_count_1000", "build_count_brick",
           "cafe_count_1500_price_high", "office_sqm_2000", "market_count_3000", "church_count_5000", "prom_part_500",
           "market_count_5000", "big_church_count_2000", "big_church_count_3000", "office_count_500",
           "detention_facility_km", "green_part_3000", "cafe_count_2000_na_price", "trc_sqm_3000", "church_count_3000",
           "church_count_1000", "culture_objects_top_25_raion", "office_sqm_500", "green_zone_part",
           "market_count_1500", "leisure_count_500", "cafe_count_3000_price_4000", "green_part_5000",
           "cafe_count_1500_na_price", "mosque_km", "kremlin_km", "thermal_power_plant_km", "big_church_count_5000",
           "school_education_centers_top_20_raion", "market_count_500", "mosque_count_5000",
           "cafe_count_1000_price_500", "market_count_1000", "mkad_km", "oil_chemistry_km",
           "cafe_count_1000_price_4000", "ttk_km", "healthcare_centers_raion", "ID_railroad_station_avto",
           "office_sqm_1000", "office_count_1500", "school_education_centers_raion", "water_treatment_km",
           "ID_bus_terminal", "cafe_count_500_price_4000", "build_count_after_1995", "ID_big_road2",
           "build_count_block", "ID_big_road1", "sport_objects_raion", "cafe_count_1500_price_4000", "office_raion",
           "build_count_slag", "university_top_20_raion", "preschool_education_centers_raion", "leisure_count_5000",
           "bulvar_ring_km", "market_count_2000", "additional_education_raion", "shopping_centers_raion",
           "ID_railroad_terminal", "cafe_count_2000_price_high"
           ]
# features =
X = X.drop(selected, axis=1)
X = X.drop(maybe_dropped, axis=1)
X = X.drop(dropped, axis=1)

# Build a forest and compute the feature importances
forest = ExtraTreesRegressor(n_estimators=250, random_state=0, n_jobs=-1)
forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%s) : %f" % (f + 1, indices[f], X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
