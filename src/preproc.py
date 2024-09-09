import pandas as pd
import numpy as np
import argparse
import io

# juice


def classify_season_slots(date):
    month = date.month
    if month in [12, 1, 2]:
        return [1, 0, 0]  # Winter
    elif month in [3, 4, 5]:
        return [0, 1, 0]  # Spring
    else:
        return [0, 0, 1]  # Summer


def classify_time_slots(time):
    hour = time.hour
    slots = [0] * 8
    index = hour // 3
    # Set the appropriate slot to 1
    slots[index] = 1
    return slots


def preprocess(
    infile, log=False, length=-1, time_features=False, season_features=False, csv=True
):
    # Load data
    if csv:
        df = pd.read_csv(infile, delimiter=";", decimal=".")
    else:
        columns = [
            "ID",
            "Data Campionamento",
            "ORA Campionamento",
            "Valore",
            "Tipo Grandezza",
        ]
        input_data = io.StringIO(infile)
        df = pd.read_csv(input_data, delimiter=";", header=None)
        df.columns = columns
    df["ID"] = df["ID"].str.replace("PDM", "").astype(int)

    # Convert datetime fields
    df["Data Campionamento"] = pd.to_datetime(df["Data Campionamento"], dayfirst=True)
    df["ORA Campionamento"] = pd.to_datetime(
        df["ORA Campionamento"], format="%H:%M:%S"
    ).dt.time

    if time_features:
        time_slots = df["ORA Campionamento"].apply(classify_time_slots)
        time_slot_df = pd.DataFrame(
            time_slots.tolist(), columns=[f"Time Slot {i}" for i in range(8)]
        )
        df = pd.concat([df, time_slot_df], axis=1)

    if season_features:
        season_slots = df["Data Campionamento"].apply(classify_season_slots)
        season_slot_df = pd.DataFrame(
            season_slots.tolist(), columns=["Winter", "Spring", "Summer"]
        )
        df = pd.concat([df, season_slot_df], axis=1)
    df_pivot = df.pivot_table(
        index=["Data Campionamento", "ORA Campionamento", "ID"],
        columns="Tipo Grandezza",
        values="Valore",
        aggfunc="first",
    ).reset_index()
    df_pivot.sort_values(
        by=["Data Campionamento", "ORA Campionamento", "ID"], inplace=True
    )
    df_pivot.reset_index(drop=True, inplace=True)

    num_ids = df["ID"].max()
    data_by_id = [None] * num_ids
    # Process each group
    for (id_index, group) in df_pivot.groupby("ID"):
        if not group.empty:
            if data_by_id[id_index - 1] is None:
                data_by_id[id_index - 1] = []
            data_by_id[id_index - 1].append(
                group[["Pressione a valle", "Temperatura Ambiente"]].to_numpy()
            )

    max_length = max(
        len(data) for sublist in data_by_id for data in sublist if sublist is not None
    )

    final_data_shape = (max_length, num_ids * 2)
    if time_features:
        final_data_shape = (
            final_data_shape[0],
            final_data_shape[1] + 8,
        )
    if season_features:
        final_data_shape = (
            final_data_shape[0],
            final_data_shape[1] + 3,
        ) 
    final_data = np.full(final_data_shape, np.nan)

    season_slot_values = season_slot_df.to_numpy() if season_features else None
    time_slot_values = time_slot_df.to_numpy() if time_features else None

    for time_step in range(max_length):
        for idx, sublist in enumerate(data_by_id):
            if sublist is not None and len(sublist[0]) > time_step:
                final_data[time_step, idx] = sublist[0][time_step, 0]
                final_data[time_step, idx + num_ids] = sublist[0][time_step, 1]
                if time_features:
                    final_data[
                        time_step, num_ids * 2 : num_ids * 2 + 8
                    ] = time_slot_values[time_step]
                if season_features:
                    final_data[time_step, -3:] = season_slot_values[time_step * 54]
    avg_pressures = np.nanmean(final_data[:, :num_ids], axis=0)
    if log:
        for i in range(len(avg_pressures)):
            print(f"ID: {i+1}, Average Pressure: {avg_pressures[i]}")
        print(avg_pressures)
    if length > 2:
        df = df.tail(length * 27)
    return final_data

def preprocess_trace(new_batch=None, infile=""):
    if new_batch is not None:
        return np.array([list(map(float, line.split(",")[:-2])) for line in new_batch])
    elif infile:
        data = np.genfromtxt(infile, delimiter=",", dtype=str)
        return data[:, :-2].astype(float)

def main():
    infile = "inputs/reversed.csv"
    df = pd.read_csv(infile, delimiter=";", decimal=".")
    dates = df["Data Campionamento"].to_numpy()
    times = df["ORA Campionamento"].to_numpy()
    indices = np.arange(0, len(dates), 54)
    times_used = times[indices]
    dates_used = dates[indices]
    data = preprocess(infile, time_features=False, season_features=False).astype(str)
    pressure = data[:, :27].astype(float)
    temperature = data[:, 27:]
    data = np.hstack((pressure, temperature, dates_used.reshape(-1, 1), times_used.reshape(-1, 1)))
    np.savetxt("inputs/traces.csv", data, delimiter=",", fmt="%s")

if __name__ == "__main__":
    main()
