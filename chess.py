import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.io as pio

# --- CONFIG ---
username = "cand5d".lower()
headers = {"User-Agent": "Mozilla/5.0"}

def fetch_bullet_games(username, year, month):
    url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}"
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        games = res.json().get("games", [])
        return [g for g in games if g.get("time_class") == "bullet"]
    except Exception as e:
        print(f"Failed to load {year}-{month:02d}: {e}")
        return []

def extract_daily_stats(games):
    daily = {}
    for g in games:
        try:
            date = datetime.utcfromtimestamp(g["end_time"]).date()
            color = "white" if g["white"]["username"].lower() == username else "black"
            result = g[color]["result"]
            win = 1 if result == "win" else 0
            daily.setdefault(date, {"games": 0, "wins": 0})
            daily[date]["games"] += 1
            daily[date]["wins"] += win
        except:
            continue

    df = pd.DataFrame([
        {"Date": date, "Games": stats["games"], "Wins": stats["wins"]}
        for date, stats in sorted(daily.items())
    ])
    if df.empty:
        print("No bullet games found.")
        return df
    df["Win Rate (%)"] = (df["Wins"] / df["Games"] * 100).round(1)
    df["Flag"] = df["Games"].apply(lambda x: "⚠️ Overuse" if x > 6 else "✅ OK")
    return df

# --- DYNAMIC MONTHS ---
today = datetime.utcnow().date()
this_month = (today.year, today.month)
last_month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)

# --- FETCH & COMBINE ---
games_combined = fetch_bullet_games(username, *last_month) + fetch_bullet_games(username, *this_month)
print(f"Fetched {len(games_combined)} games")
df = extract_daily_stats(games_combined)

# --- PLOT ---
if not df.empty:
    fig = go.Figure(data=[
        go.Bar(
            x=df["Date"], y=df["Games"],
            name="Games per Day",
            marker_color=['red' if x > 6 else 'green' for x in df["Games"]],
            text=df["Games"], textposition="auto",
            yaxis="y1"
        ),
        go.Scatter(
            x=df["Date"], y=df["Win Rate (%)"],
            name="Win Rate (%)",
            mode='lines+markers',
            line=dict(color="blue"),
            marker=dict(size=10),
            yaxis="y2"
        )
    ])
    fig.update_layout(
        title="Bullet Games and Win Rate per Day",
        height=500,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Games", side="left"),
        yaxis2=dict(title="Win Rate (%)", side="right", overlaying="y"),
        legend=dict(x=0.1, y=1.1, orientation="h")
    )
else:
    fig = go.Figure()
    fig.update_layout(title="No Bullet Games Found", height=500)

# --- GENERATE HTML ---
if not df.empty:
    df_table = df.sort_values('Date', ascending=False)
    table_html = df_table.to_html(index=False)
else:
    table_html = '<p>No data available.</p>'

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bullet Chess Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Bullet Chess Dashboard – Last 2 Months</h1>
    {pio.to_html(fig, full_html=False, include_plotlyjs='cdn')}
    <h3>Daily Summary Table</h3>
    {table_html}
</body>
</html>
"""
with open("index.html", "w") as f:
    f.write(html_content)
print("Generated index.html")
