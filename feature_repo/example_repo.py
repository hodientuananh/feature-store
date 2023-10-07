# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Int64

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
driver = Entity(name="ConsumerType_DE35", join_keys=["ConsumerType_DE35_id"])

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
driver_stats_source = FileSource(
    name="hourly_energy_consumption",
    path="/root/assignment/feature_repo/data/data_for_project.parquet",
    timestamp_field="HourUTC",
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
driver_stats_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="hourly_energy_consumption",
    # entities=[driver],
    ttl=timedelta(days=1),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="PriceArea", dtype=Int64),
        Field(name="ConsumerType_DE35", dtype=Int64),
        Field(name="TotalCon", dtype=Int64),
        Field(name="HourUTC_year", dtype=Int64),
        Field(name="HourUTC_month", dtype=Int64),
        Field(name="HourUTC_day", dtype=Int64),
        Field(name="HourDK_year", dtype=Int64),
        Field(name="HourDK_month", dtype=Int64),
        Field(name="HourDK_day", dtype=Int64),
    ],
    online=True,
    source=driver_stats_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "hourly_consumption"},
)

# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
input_request = RequestSource(
    name="vals_to_add",
    schema=[
        Field(name="val_to_add", dtype=Int64),
        Field(name="val_to_add_2", dtype=Int64),
    ],
)


# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
@on_demand_feature_view(
    sources=[driver_stats_fv, input_request],
    schema=[
        Field(name="TotalCon_plus_val1", dtype=Int64),
        Field(name="TotalCon_plus_val2", dtype=Int64),
    ],
)
def transformed_TotalCon(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["TotalCon_plus_val1"] = inputs["TotalCon"] + inputs["val_to_add"]
    df["TotalCon_plus_val2"] = inputs["TotalCon"] + inputs["val_to_add_2"]
    return df


# This groups features into a model version
driver_activity_v1 = FeatureService(
    name="driver_activity_v1",
    features=[
        driver_stats_fv[["TotalCon"]],  # Sub-selects a feature from a feature view
        transformed_TotalCon,  # Selects all features from the feature view
    ],
)
driver_activity_v2 = FeatureService(
    name="driver_activity_v2", features=[driver_stats_fv, transformed_TotalCon]
)

# Defines a way to push data (to be available offline, online or both) into Feast.
driver_stats_push_source = PushSource(
    name="driver_stats_push_source",
    batch_source=driver_stats_source,
)

# Defines a slightly modified version of the feature view from above, where the source
# has been changed to the push source. This allows fresh features to be directly pushed
# to the online store for this feature view.
driver_stats_fresh_fv = FeatureView(
    name="hourly_energy_consumption_fresh",
    # entities=[driver],
    ttl=timedelta(days=1),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="PriceArea", dtype=Int64),
        Field(name="ConsumerType_DE35", dtype=Int64),
        Field(name="TotalCon", dtype=Int64),
        Field(name="HourUTC_year", dtype=Int64),
        Field(name="HourUTC_month", dtype=Int64),
        Field(name="HourUTC_day", dtype=Int64),
        Field(name="HourDK_year", dtype=Int64),
        Field(name="HourDK_month", dtype=Int64),
        Field(name="HourDK_day", dtype=Int64),
    ],
    online=True,
    source=driver_stats_push_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "hourly_consumption"},
)


# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
@on_demand_feature_view(
    sources=[driver_stats_fresh_fv, input_request],  # relies on fresh version of FV
    schema=[
        Field(name="TotalCon_plus_val1", dtype=Int64),
        Field(name="TotalCon_plus_val2", dtype=Int64),
    ],
)
def transformed_TotalCon_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["TotalCon_plus_val1"] = inputs["TotalCon"] + inputs["val_to_add"]
    df["TotalCon_plus_val2"] = inputs["TotalCon"] + inputs["val_to_add_2"]
    return df


driver_activity_v3 = FeatureService(
    name="driver_activity_v3",
    features=[driver_stats_fresh_fv, transformed_TotalCon_fresh],
)
