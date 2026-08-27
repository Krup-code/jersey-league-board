"""Rebuilds index.html from fresh data. Run daily by .github/workflows/refresh.yml."""
import html as htmlmod
import json
import math
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from ffdraft.config import LeagueSettings, ModelWeights, Scoring
from ffdraft.model import build_player_table, project
from ffdraft.board import attach_adp, load_adp

TEMPLATE_PATH = "index_template.html"
OUTPUT_PATH = "index.html"

LEAGUE = LeagueSettings(
    name="jersey league",
    teams=12,
    rounds=17,
    draft_slot=9,
    scoring=Scoring.preset("ppr"),
    starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DST": 1},
)


def r(x, n=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), n)


def build_board_rows():
    weights = ModelWeights()
    tbl = build_player_table(LEAGUE, weights, season=2026)
    board = project(tbl, LEAGUE, weights)
    adp = load_adp(fmt="ppr", season=2026)
    board = attach_adp(board, adp)

    rows = []
    for _, p in board.sort_values("overall_rank").iterrows():
        rows.append({
            "rk": int(p["overall_rank"]),
            "pr": f'{p["position"]}{int(p["pos_rank"]) if not math.isnan(p["pos_rank"]) else ""}',
            "n": p["name"], "pos": p["position"],
            "tm": p["team"] if isinstance(p["team"], str) else "-",
            "adp": r(p["adp"], 1), "adpd": r(p["adp_delta"], 1), "pp": r(p["proj_points_ppr"], 0),
            "vor": r(p["vor"], 0), "cons": r(p["consistency"], 3), "age": r(p["age"], 1),
            "gm": r(p["exp_games"], 1), "inj": r(p["injury_risk"], 3), "rook": bool(p["is_rookie"]),
            "ol": int(p["run_block_rank"]) if p["position"] == "RB" and not math.isnan(p.get("run_block_rank", float("nan")))
                  else (int(p["pass_block_rank"]) if not math.isnan(p.get("pass_block_rank", float("nan"))) else None),
            "ppg": r(p["plays_per_game"], 0), "sos": r(p.get(f"sos_{p['position']}_z", float("nan")), 2),
            "yprr": r(p["yprr"], 2), "tprr": r(p["tprr"], 3), "sep": r(p["avg_separation"], 1),
            "rztd": r(p["rz_td_rate"], 2) if p["rz_touches"] and p["rz_touches"] >= 5 else None,
            "floor": r(p["floor"], 1), "ceil": r(p["ceiling"], 1), "fpm": r(p["fp_mean"], 1),
        })
    return rows


def build_lineups():
    dc = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.parquet")
    latest = dc["dt"].max()
    dc = dc[dc["dt"] == latest]
    dc = dc[dc["pos_abb"].isin(["QB", "RB", "WR", "TE", "FB"])]
    dc = dc.dropna(subset=["player_name"])
    dc = dc.sort_values(["team", "pos_abb", "pos_rank"])
    dc = dc.groupby(["team", "pos_abb"]).head(4)

    lineups = {}
    for team, grp in dc.groupby("team"):
        lineups[team] = {}
        for pos, pgrp in grp.groupby("pos_abb"):
            lineups[team][pos] = [
                {"rk": int(row["pos_rank"]), "n": row["player_name"]}
                for _, row in pgrp.sort_values("pos_rank").iterrows()
            ]
    return latest, lineups


def build_schedule_and_byes():
    from ffdraft import sources
    sc = sources.schedules()
    sc = sc[(sc["season"] == 2026) & (sc["game_type"] == "REG")].copy()

    games = []
    for _, rr in sc.sort_values(["week", "gameday", "gametime"]).iterrows():
        games.append({
            "wk": int(rr["week"]), "d": rr["gameday"], "wd": rr["weekday"], "t": rr["gametime"],
            "away": rr["away_team"], "home": rr["home_team"], "div": bool(rr["div_game"]),
        })

    all_teams = sorted(set(sc["home_team"]) | set(sc["away_team"]))
    byes = {}
    for team in all_teams:
        played = set(sc[(sc["home_team"] == team) | (sc["away_team"] == team)]["week"])
        bye = sorted(set(range(1, 19)) - played)
        byes[team] = bye[0] if bye else None
    return games, byes


def clean_html(s):
    if not s:
        return ""
    s = htmlmod.unescape(s)
    return re.sub(r"<[^>]+>", "", s).strip()


def build_news():
    resp = requests.get("https://www.espn.com/espn/rss/nfl/news", timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    news = []
    for item in root.findall(".//item"):
        news.append({
            "t": clean_html(item.findtext("title")),
            "d": clean_html(item.findtext("description")),
            "l": (item.findtext("link") or "").strip(),
            "p": (item.findtext("pubDate") or "").strip(),
            "src": clean_html(item.findtext("{http://purl.org/dc/elements/1.1/}creator")) or "ESPN",
        })
    return news


def main():
    rows = build_board_rows()
    print(f"board rows: {len(rows)}")

    lineups_asof, lineups = build_lineups()
    print(f"lineup teams: {len(lineups)} (as of {lineups_asof})")

    games, byes = build_schedule_and_byes()
    print(f"scheduled games: {len(games)}")

    for p in rows:
        p["bye"] = byes.get(p["tm"])

    news = build_news()
    print(f"news items: {len(news)}")

    with open(TEMPLATE_PATH) as f:
        html = f.read()

    payloads = {
        "__BOARD_DATA__": json.dumps(rows, separators=(",", ":"), allow_nan=False),
        "__LINEUPS_DATA__": json.dumps({"asOf": lineups_asof, "teams": lineups}, separators=(",", ":"), allow_nan=False),
        "__SCHEDULE_DATA__": json.dumps({"games": games, "byes": byes}, separators=(",", ":"), allow_nan=False),
        "__NEWS_DATA__": json.dumps(news, separators=(",", ":"), allow_nan=False),
    }
    for tag, payload in payloads.items():
        assert html.count(tag) == 1, f"expected exactly one {tag} in template"
        html = html.replace(tag, payload)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUTPUT_PATH}, {len(html)} bytes")


if __name__ == "__main__":
    main()
