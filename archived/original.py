import json
import time
from os import environ
from os.path import expanduser
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


def get_datafields(
    s,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    dataset_id: str = "",
    data_type: str = "MATRIX",
    search: str = "",
):
    offset = 0
    datafields_list = []

    while True:
        url_template = (
            "https://api.worldquantbrain.com/data-fields?"
            + f"&instrumentType={instrument_type}"
            + f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50"
            + f"&offset={offset}"
            + f"&type={data_type}"
        )
        url_template += f"&search={search}" if search else ""
        resp = s.get(url_template)
        results = resp.json()

        if "results" not in results:
            print(f"Unexpected response: {results}")
            break
        else:
            print(f"Fetched {len(results['results'])} data fields with offset {offset}.")
            datafields_list.append(results["results"])
            if len(results["results"]) < 50:
                print("Fetched the last batch of data fields.")
                break
            offset += 50
            time.sleep(5)

    datafields_list_flat = [item for sublist in datafields_list for item in sublist]
    return pd.DataFrame(datafields_list_flat)


def main():
    # Load credentials
    credentials = None
    cred_file = Path(expanduser("brain_credentials.txt"))
    if cred_file.exists():
        with open(cred_file) as f:
            credentials = json.load(f)
    elif (Path(__file__).resolve().parent.parent / ".env").exists():
        dotenv = Path(__file__).resolve().parent.parent / ".env"
        env_vars = {}
        with open(dotenv) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
        credentials = (env_vars.get("BRAIN_USERNAME"), env_vars.get("BRAIN_PASSWORD"))
    else:
        credentials = (environ.get("BRAIN_USERNAME"), environ.get("BRAIN_PASSWORD"))

    username, password = credentials
    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)
    response = sess.post("https://api.worldquantbrain.com/authentication")
    print(response.status_code)
    print(response.json())

    fundamental6 = get_datafields(s=sess, dataset_id="pv13", data_type="GROUP")
    if "id" not in fundamental6.columns:
        print("No 'id' column found in fetched data fields.")
        return

    datafields_list = fundamental6["id"].values
    print(f"Total data fields: {len(datafields_list)}")

    alpha_list = []
    group_ops_list = ["group_mean", "group_neutralize"]
    ts_ops_list = ["ts_mean", "ts_rank"]
    days = [63, 126]
    groups = ["market", "sector", "industry"]

    for datafield in datafields_list:
        for group_ops in group_ops_list:
            for ts_ops in ts_ops_list:
                for day in days:
                    for group in groups:
                        print("Packaging the following Alpha expression with settings")
                        print(f"{group_ops}({ts_ops}({datafield}, {day}), {group})")
                        simulation_data = {
                            "type": "REGULAR",
                            "settings": {
                                "instrumentType": "EQUITY",
                                "region": "USA",
                                "universe": "TOP3000",
                                "delay": 1,
                                "decay": 0,
                                "neutralization": "MARKET",
                                "truncation": 0.08,
                                "pasteurization": "ON",
                                "unitHandling": "VERIFY",
                                "nanHandling": "ON",
                                "language": "FASTEXPR",
                                "visualization": False,
                            },
                            "regular": f"{group_ops}({ts_ops}({datafield}, {day}), {group})",
                        }
                        alpha_list.append(simulation_data)

    for alpha in alpha_list:
        sim_resp = sess.post(
            "https://api.worldquantbrain.com/simulations",
            json=alpha,
        )
        try:
            sim_progress_url = sim_resp.headers["Location"]
            while True:
                sim_progress_resp = sess.get(sim_progress_url)
                retry_after_sec = float(sim_progress_resp.headers.get("Retry-After", 0))
                if retry_after_sec == 0:
                    break
                time.sleep(retry_after_sec)
            alpha_id = sim_progress_resp.json()["alpha"]
            print(alpha_id)
        except Exception:
            print("no location, sleep for 10 seconds and try next alpha")
            time.sleep(10)


if __name__ == "__main__":
    main()
