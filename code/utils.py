import pandas as pd 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from pathlib import Path
import json
import requests
from urllib.request import urlopen
import plotly.io as pio
import os
from ipumspy import IpumsApiClient, MicrodataExtract, readers, ddi
import plotly.graph_objs as go
import pymc as pm
import arviz as az
import re
import pymc as pm
from scipy.special import logit
import xarray as xr
import arviz as az

def census_json_request(url,params,timeout=60):
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json()

def get_county_female_occupation_counts(table_id, census_key):
    base_url = (f"https://api.census.gov/data/2022/acs/acs5/subject")

    metadata_params = {}
    metadata_params["key"] = census_key

    response = requests.get(f"{base_url}/groups/{table_id}.json" ,params=metadata_params,timeout=30)
    response.raise_for_status()
    metadata = response.json()

    all_variables = metadata["variables"]
    female_variables = {
        variable: details["label"]
        for variable, details in all_variables.items()
        if variable.startswith(f"{table_id}_")
        and variable.endswith("E")
        and details.get("label", "").startswith("Estimate!!Female!!")
    }

    def variable_number(variable):
        match = re.search(r"_(\d{3})E$", variable)
        return int(match.group(1)) if match else 9999

    female_codes = sorted(
        female_variables,
        key=variable_number,
    )

    total_female_code = f"{table_id}_C04_001E"

    requested_variables = ["NAME", *female_codes]

    state_data = census_json_request(
        url=base_url,
        params={ "get": "NAME",
        "for": "state:*",
        "key":census_key})

    states = pd.DataFrame(state_data[1:],columns=state_data[0])
    states = states.loc[states["state"] != "72"].copy()

    county_frames = []

    for state_fips in states["state"]:
        data = census_json_request(
            url=base_url,
            params={
            "get": ",".join(requested_variables),
            "for": "county:*",
            "in": f"state:{state_fips}",
            "key":census_key})

        county_frames.append(pd.DataFrame(data[1:],columns=data[0]))

    occupation_df = pd.concat(county_frames,ignore_index=True)

    # replace values below 0 with blanks
    occupation_df[female_codes] = occupation_df[female_codes].apply(pd.to_numeric,errors="coerce")
    occupation_df[female_codes] = occupation_df[female_codes].mask(occupation_df[female_codes] < 0)

    occupation_df["county_fips"] = (occupation_df["state"].str.zfill(2)+ occupation_df["county"].str.zfill(3))

    occupation_df = occupation_df.rename(columns={total_female_code: "female_employed_total"})

    renamed_female_codes = ["female_employed_total" if code == total_female_code else code for code in female_codes]

    for original_code, current_code in zip(female_codes, renamed_female_codes):
        if current_code == "female_employed_total":
            continue
        occupation_df[f"{current_code}_share"] = occupation_df[current_code] / occupation_df["female_employed_total"].replace(0, pd.NA)


    variable_dictionary = pd.DataFrame({"variable": female_codes,"label": [female_variables[code] for code in female_codes]})
    variable_dictionary["occupation"] = (variable_dictionary["label"].str.split("!!").str[-1].str.rstrip(":"))
    variable_dictionary["is_total"] = (variable_dictionary["variable"]== total_female_code)

    return occupation_df, variable_dictionary


def create_summary_table(idata, parameter_names):

    summary_table = az.summary(
        idata,
        group="posterior",
        var_names=list(parameter_names.keys()),
        ci_prob=0.89,
        ci_kind="eti",
        round_to="none",
    )

    summary_table = (
        summary_table
        .rename(index=parameter_names)
        .rename(
            columns={
                "mean": "Posterior mean",
                "sd": "Posterior SD",
                "eti89_lb": "89% credible interval lower",
                "eti89_ub": "89% credible interval upper",
                "ess_bulk": "Bulk ESS",
                "ess_tail": "Tail ESS",
                "r_hat": "R-hat",
            }
        )
    )

    summary_table.index.name = "Parameter"

    report_table = summary_table[
        [
            "Posterior mean",
            "Posterior SD",
            "89% credible interval lower",
            "89% credible interval upper",
            "R-hat",
        ]
    ].astype(float).round(3)

    return report_table