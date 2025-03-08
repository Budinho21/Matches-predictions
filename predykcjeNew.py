import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime, timedelta


# --- 1. POBIERANIE DANYCH HISTORYCZNYCH ---
def fetch_historical_data():
    print("Pobieranie danych historycznych...")
    leagues = {
        "Premier League": "PL",
        "Bundesliga": "BL1",
        "La Liga": "PD",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Eredivisie": "DED",
        "Primeira Liga": "PPL"
    }

    all_matches = []
    seasons = [2023, 2024]  # Lista lat, z których będziemy pobierać dane

    for league_name, code in leagues.items():
        for season in seasons:
            url = f"http://api.football-data.org/v4/competitions/{code}/matches?season={season}"
            headers = {"X-Auth-Token": "a1c9066c65b74dd5b802b982485a0a81"}  # Wstaw swój klucz API
            response = requests.get(url, headers=headers)

            print(f"Żądanie dla {league_name}, sezon {season}: {url}")  # Komunikat diagnostyczny

            if response.status_code == 200:
                data = response.json()
                print(f"Dane dla {league_name}, sezon {season}: {data}")  # Komunikat diagnostyczny
                print(f"Liczba meczów: {len(data['matches'])}")

                for match in data['matches']:
                    if match['status'] == 'FINISHED':
                        home_goals = match['score']['fullTime']['home']
                        away_goals = match['score']['fullTime']['away']
                        total_goals = home_goals + away_goals
                        all_matches.append([  # Dodajemy mecz do listy
                            league_name,
                            match['homeTeam']['name'],
                            match['awayTeam']['name'],
                            home_goals,
                            away_goals,
                            total_goals,
                            1 if total_goals > 2.5 else 0  # Over 2.5
                        ])
            else:
                print(f"Nie udało się pobrać danych dla {league_name}, sezon: {season}")

    df = pd.DataFrame(all_matches, columns=[  # Kolumny danych historycznych
        "League", "Home Team", "Away Team", "Home Goals", "Away Goals", "Total Goals", "Over 2.5"
    ])

    # Sprawdzamy, jakie ligi są w DataFrame po pobraniu danych
    print(f"Ligi w DataFrame: {df['League'].unique()}")
    return df


# --- 2. PRZETWARZANIE DANYCH ---
def prepare_features(df, le=None):
    print("Przygotowywanie cech...")

    if le is None:
        le = LabelEncoder()
        df["Home Team"] = le.fit_transform(df["Home Team"])
        df["Away Team"] = le.fit_transform(df["Away Team"])
    else:
        df["Home Team"] = le.transform(df["Home Team"])
        df["Away Team"] = le.transform(df["Away Team"])

    # Sprawdzamy dostępność kolumny 'Average Goals' w danych
    if 'Average Goals (5)' not in df.columns or 'Average Goals (30)' not in df.columns:
        print("Kolumna 'Average Goals (5)' lub 'Average Goals (30)' nie istnieje. Obliczamy średnią bramek...")

        # Obliczamy średnią bramek z ostatnich 5 meczów
        df["Average Goals (5)"] = df["Total Goals"].rolling(window=5).mean().fillna(2.5)

        # Obliczamy średnią bramek z ostatnich 30 meczów
        df["Average Goals (30)"] = df["Total Goals"].rolling(window=30).mean().fillna(2.5)

    # Obliczamy odchylenie pomiędzy średnimi z 5 i 30 meczów
    df["Deviation"] = abs(df["Average Goals (5)"] - df["Average Goals (30)"])

    # Tworzymy nową kolumnę 'Final Average Goals' bazując na wagi średnich
    df["Final Average Goals"] = np.where(
        df["Deviation"] > 0.5,  # Jeśli odchylenie większe niż 0.5
        (0.7 * df["Average Goals (5)"]) + (0.3 * df["Average Goals (30)"]),  # Większa waga dla window=5
        (0.3 * df["Average Goals (5)"]) + (0.7 * df["Average Goals (30)"])  # Większa waga dla window=30
    )

    df["Is Over 2.5"] = df["Over 2.5"]
    df = df.drop(columns=["Home Goals", "Away Goals", "Total Goals", "Over 2.5", "League",
                          "Deviation"])  # Usuwamy zbędne kolumny

    return df, le


# --- 3. TRENING MODELU ---
def train_model(df):
    print("Trenowanie modelu...")
    X = df.drop(columns=["Is Over 2.5"])
    y = df["Is Over 2.5"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Dokładność modelu: {accuracy * 100:.2f}%")

    return model


# --- 4. POBRANIE NADCHODZĄCEJ MECZY ---
def filter_upcoming_matches(upcoming_matches):
    today = datetime.today()
    three_days_later = today + timedelta(days=3)

    # Konwertujemy daty w DataFrame na obiekty datetime
    upcoming_matches["Match Date"] = pd.to_datetime(upcoming_matches["Match Date"])

    # Filtrujemy mecze, które są w zakresie dzisiaj - 3 dni później
    filtered_matches = upcoming_matches[
        (upcoming_matches["Match Date"] >= today) &
        (upcoming_matches["Match Date"] <= three_days_later)
        ]

    return filtered_matches


def fetch_upcoming_matches():
    leagues = {
        "Premier League": "PL",
        "Bundesliga": "BL1",
        "La Liga": "PD",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Eredivisie": "DED",
        "Primeira Liga": "PPL"
    }

    upcoming_matches = []

    for league_name, code in leagues.items():
        url = f"http://api.football-data.org/v4/competitions/{code}/matches?status=SCHEDULED"
        headers = {
            "X-Auth-Token": "a1c9066c65b74dd5b802b982485a0a81",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
            "Accept": "application/json"
        }

        print(f"Żądanie dla nadchodzących meczów {league_name}: {url}")

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 429:  # Jeśli przekroczono limit zapytań
                wait_time = int(response.headers.get('Retry-After', 60))  # Pobieramy czas oczekiwania z nagłówka
                print(f"Limit zapytań przekroczony, czekam {wait_time} sekund...")
                time.sleep(wait_time)  # Czekaj określoną ilość czasu
                response = requests.get(url, headers=headers)  # Spróbuj ponownie

            if response.status_code == 200:
                data = response.json()
                for match in data['matches']:
                    upcoming_matches.append([  # Dodajemy tylko istotne informacje
                        league_name,
                        match['homeTeam']['name'],
                        match['awayTeam']['name'],
                        match['utcDate']  # Dodajemy datę przewidywanego meczu
                    ])
            else:
                print(f"Nie udało się pobrać nadchodzących meczów dla {league_name}, status: {response.status_code}")
                print(f"Treść odpowiedzi: {response.text}")
        except Exception as e:
            print(f"Wystąpił błąd podczas pobierania danych dla {league_name}: {e}")

    # Tworzymy DataFrame z datą meczu
    df = pd.DataFrame(upcoming_matches, columns=["League", "Home Team", "Away Team", "Match Date"])

    # Zmieniamy format daty, jeśli jest potrzeba
    df["Match Date"] = pd.to_datetime(df["Match Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Filtrujemy mecze tylko na nadchodzące 3 dni
    filtered_matches = filter_upcoming_matches(df)

    # Zapisujemy tylko przefiltrowane mecze do pliku CSV
    filtered_matches.to_csv("predictions.csv", index=False)
    print(f"Plik predictions.csv zapisany dla meczów na nadchodzące 3 dni.")

    return filtered_matches


def compute_team_averages(historical_data, window):
    team_avg_goals = {}

    for team in set(historical_data["Home Team"]).union(set(historical_data["Away Team"])):

        # Ostatnie mecze drużyny jako gospodarza
        home_matches = historical_data[historical_data["Home Team"] == team]["Total Goals"].tail(window)

        # Ostatnie mecze drużyny jako gościa
        away_matches = historical_data[historical_data["Away Team"] == team]["Total Goals"].tail(window)

        # Liczymy średnią łącznej liczby bramek (strzelone + stracone)
        avg_goals = pd.concat([home_matches, away_matches]).mean()

        if not np.isnan(avg_goals):
            team_avg_goals[team] = avg_goals
        else:
            # Jeśli brak danych, można przypisać np. średnią z całego zestawu meczów
            team_avg_goals[team] = historical_data["Total Goals"].mean()

    return team_avg_goals


def predict_over_2_5(model, upcoming_matches, le, historical_data):
    print("Przewidywanie wyników dla nadchodzących meczów...")

    # Sprawdzenie, które drużyny są nowe
    new_teams = set(upcoming_matches["Home Team"]).union(set(upcoming_matches["Away Team"])) - set(le.classes_)
    if new_teams:
        print(f"Znaleziono nowe drużyny: {new_teams}")
        le.fit(list(le.classes_) + list(new_teams))

    # Tworzymy kopię danych, aby uniknąć ostrzeżeń o kopii
    upcoming_matches = upcoming_matches.copy()

    upcoming_matches["Home Team"] = le.transform(upcoming_matches["Home Team"])
    upcoming_matches["Away Team"] = le.transform(upcoming_matches["Away Team"])

    # Obliczanie średnich bramek na podstawie historycznych danych
    avg_goals_5 = compute_team_averages(historical_data, 5)
    avg_goals_30 = compute_team_averages(historical_data, 30)

    # Przypisanie średnich bramek tylko drużynom, które mają dane
    upcoming_matches["Average Goals (5)"] = upcoming_matches["Home Team"].map(avg_goals_5) + upcoming_matches["Away Team"].map(avg_goals_5)
    upcoming_matches["Average Goals (30)"] = upcoming_matches["Home Team"].map(avg_goals_30) + upcoming_matches["Away Team"].map(avg_goals_30)

    # Usunięcie meczów, w których którakolwiek drużyna nie ma średniej liczby bramek
    upcoming_matches = upcoming_matches.dropna(subset=["Average Goals (5)", "Average Goals (30)"])

    # Obliczanie "Deviation" i "Final Average Goals"
    upcoming_matches.loc[:, "Deviation"] = abs(upcoming_matches["Average Goals (5)"] - upcoming_matches["Average Goals (30)"])

    upcoming_matches.loc[:, "Final Average Goals"] = np.where(
        upcoming_matches["Deviation"] > 0.5,
        (0.7 * upcoming_matches["Average Goals (5)"]) + (0.3 * upcoming_matches["Average Goals (30)"]),
        (0.3 * upcoming_matches["Average Goals (5)"]) + (0.7 * upcoming_matches["Average Goals (30)"])
    )

    # Przygotowanie cech do modelu
    X = upcoming_matches[["Home Team", "Away Team", "Average Goals (5)", "Average Goals (30)", "Final Average Goals"]]

    # Przewidywanie Over 2.5
    predictions_proba = model.predict_proba(X)
    upcoming_matches.loc[:, "Prediction Over 2.5"] = predictions_proba[:, 1] * 100
    upcoming_matches.loc[:, "Prediction Over 2.5"] = upcoming_matches["Prediction Over 2.5"].round(2).astype(str) + "%"

    # Przywrócenie nazw drużyn
    upcoming_matches.loc[:, "Home Team"] = le.inverse_transform(upcoming_matches["Home Team"])
    upcoming_matches.loc[:, "Away Team"] = le.inverse_transform(upcoming_matches["Away Team"])

    # Dodanie minimalnego kursu
    upcoming_matches.loc[:, "Minimum Odd"] = (1 / (predictions_proba[:, 1] * 0.88)) + 0.01
    upcoming_matches.loc[:, "Minimum Odd"] = upcoming_matches["Minimum Odd"].round(2)

    # Zapis wyników
    upcoming_matches.to_csv("predictions.csv", index=False)
    print("Plik predictions.csv zapisany z przewidywaniami szansy na Over 2.5.")

    return upcoming_matches

# --- 6. GŁÓWNA FUNKCJA ---
def main():
    historical_data = fetch_historical_data()
    processed_data, le = prepare_features(historical_data)
    model = train_model(processed_data)

    upcoming_matches = fetch_upcoming_matches()
    predictions = predict_over_2_5(model, upcoming_matches, le, historical_data)

    return predictions

if __name__ == "__main__":
    main()

# DO ZEROBIENIA:
# funkcja predict_over_2_5 - zweryfikowanei i ustaleniu sposobu obliczania średniej dla 5 i 30 meczów, żeby nie była wprowadzana stała 2.5