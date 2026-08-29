"""Rebuilds index.html from fresh data. Run daily by .github/workflows/refresh.yml."""
import html as htmlmod
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

from ffdraft.config import LeagueSettings, ModelWeights, Scoring
from ffdraft.model import build_player_table, project
from ffdraft.board import attach_adp, load_adp
from ffdraft import features, sources

TEMPLATE_PATH = "index_template.html"
OUTPUT_PATH = "index.html"

OFFENSE_ORDER = ["QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT"]
DEFENSE_ORDER = ["LDE", "LDT", "NT", "RDT", "RDE", "WLB", "MLB", "SLB", "LILB", "RILB",
                  "LCB", "RCB", "NB", "SS", "FS"]
SPECIAL_ORDER = ["PK", "P", "H", "LS", "PR", "KR"]
UNIT_OF = ({p: "OFF" for p in OFFENSE_ORDER}
           | {p: "DEF" for p in DEFENSE_ORDER}
           | {p: "ST" for p in SPECIAL_ORDER})

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
    """Full offense/defense/special-teams depth chart, every position on the field."""
    dc = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.parquet")
    latest = dc["dt"].max()
    dc = dc[dc["dt"] == latest]
    dc = dc[dc["pos_abb"].isin(list(UNIT_OF.keys()))]
    dc = dc.dropna(subset=["player_name"])
    dc = dc.sort_values(["team", "pos_abb", "pos_rank"])
    dc = dc.groupby(["team", "pos_abb"]).head(3)

    lineups = {}
    for team, grp in dc.groupby("team"):
        lineups[team] = {"OFF": {}, "DEF": {}, "ST": {}}
        for pos, pgrp in grp.groupby("pos_abb"):
            unit = UNIT_OF[pos]
            lineups[team][unit][pos] = [
                {"rk": int(row["pos_rank"]), "n": row["player_name"]}
                for _, row in pgrp.sort_values("pos_rank").iterrows()
            ]
    return latest, lineups


def implied_prob(moneyline):
    if moneyline is None or (isinstance(moneyline, float) and math.isnan(moneyline)):
        return None
    return (-moneyline / (-moneyline + 100)) if moneyline < 0 else (100 / (moneyline + 100))


def build_schedule_and_byes():
    sc = sources.schedules()
    sc = sc[(sc["season"] == 2026) & (sc["game_type"] == "REG")].copy()

    games = []
    for _, rr in sc.sort_values(["week", "gameday", "gametime"]).iterrows():
        home_p = implied_prob(rr["home_moneyline"])
        away_p = implied_prob(rr["away_moneyline"])
        hwp = None
        if home_p is not None and away_p is not None and (home_p + away_p) > 0:
            hwp = round(100 * home_p / (home_p + away_p))
        games.append({
            "wk": int(rr["week"]), "d": rr["gameday"], "wd": rr["weekday"], "t": rr["gametime"],
            "away": rr["away_team"], "home": rr["home_team"], "div": bool(rr["div_game"]),
            "hwp": hwp,
            "spread": r(rr["spread_line"], 1),
            "total": r(rr["total_line"], 1),
        })

    all_teams = sorted(set(sc["home_team"]) | set(sc["away_team"]))
    byes = {}
    for team in all_teams:
        played = set(sc[(sc["home_team"] == team) | (sc["away_team"] == team)]["week"])
        bye = sorted(set(range(1, 19)) - played)
        byes[team] = bye[0] if bye else None
    return games, byes


def build_team_stats():
    """Real 2025 final standings (last completed season) plus the model's own
    O-line/pace/defense ratings, computed from raw plays rather than a ranking site."""
    pbp = sources.play_by_play(seasons=[2021, 2022, 2023, 2024, 2025])
    ol = features.oline_ratings(pbp)
    pace = features.team_pace_and_split(pbp)
    dfn = features.defense_ratings(pbp, sources.weekly_stats([2021, 2022, 2023, 2024, 2025]), Scoring(rec=1.0))
    recent = int(pace["season"].max())
    ol_r = ol[ol["season"] == recent].set_index("team")
    pace_r = pace[pace["season"] == recent].set_index("team")
    dfn_r = dfn[dfn["season"] == recent].set_index("team")

    sc = sources.schedules()
    d = sc[(sc["season"] == recent) & (sc["game_type"] == "REG") & sc["home_score"].notna()]
    standings = {}
    for _, g in d.iterrows():
        for team, pf, pa in ((g["home_team"], g["home_score"], g["away_score"]),
                              (g["away_team"], g["away_score"], g["home_score"])):
            s = standings.setdefault(team, {"w": 0, "l": 0, "t": 0, "pf": 0.0, "pa": 0.0, "g": 0})
            s["g"] += 1
            s["pf"] += pf
            s["pa"] += pa
            if pf > pa:
                s["w"] += 1
            elif pf < pa:
                s["l"] += 1
            else:
                s["t"] += 1

    out = []
    for team, s in standings.items():
        row = {
            "tm": team, "season": recent, "w": s["w"], "l": s["l"], "t": s["t"],
            "pf": r(s["pf"] / s["g"], 1), "pa": r(s["pa"] / s["g"], 1),
        }
        if team in pace_r.index:
            row["ppg"] = r(pace_r.loc[team, "plays_per_game"], 0)
            row["pass_pct"] = r(pace_r.loc[team, "pass_rate"] * 100, 0)
        if team in ol_r.index:
            row["rbk"] = int(ol_r.loc[team, "run_block_rank"])
            row["pbk"] = int(ol_r.loc[team, "pass_block_rank"])
        if team in dfn_r.index:
            row["drk"] = int(dfn_r.loc[team, "def_rank"]) if not pd.isna(dfn_r.loc[team, "def_rank"]) else None
            fpa = {}
            for pos in ("QB", "RB", "WR", "TE"):
                col = f"fpa_{pos}_rank"
                if col in dfn_r.columns and not pd.isna(dfn_r.loc[team, col]):
                    fpa[pos] = int(dfn_r.loc[team, col])
            row["fpa"] = fpa
        out.append(row)
    return sorted(out, key=lambda x: (-x["w"], x["l"]))


def clean_html(s):
    if not s:
        return ""
    s = htmlmod.unescape(s)
    return re.sub(r"<[^>]+>", "", s).strip()


def previous_news():
    """Best-effort fallback: pull the news payload out of the last built index.html.

    Some hosts (ESPN included) rate-limit or block requests from cloud/CI IP ranges
    even with a normal User-Agent, which is IP-reputation based rather than anything
    fixable in the request itself. Rather than fail the whole refresh -- or silently
    wipe the News tab -- fall back to yesterday's real headlines so the page still
    shows something genuine, just possibly a day stale.
    """
    try:
        with open(OUTPUT_PATH) as f:
            html = f.read()
        m = re.search(r'<script id="news-data"[^>]*>(.*?)</script>', html, re.S)
        return json.loads(m.group(1)) if m else []
    except Exception:
        return []


def build_team_logos():
    """Official team logo/color data from nflverse's community-maintained team manifest --
    same lineage as every other data source on this site, not hand-picked links."""
    current = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
               "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
               "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
               "TEN", "WAS"]
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/nflverse/nflfastR-data/master/teams_colors_logos.csv")
        d = df[df["team_abbr"].isin(current)].set_index("team_abbr")
        return {t: {"logo": d.loc[t, "team_logo_espn"], "c1": d.loc[t, "team_color"], "c2": d.loc[t, "team_color2"]}
                for t in current if t in d.index}
    except Exception as exc:
        print(f"team logo fetch failed ({type(exc).__name__}: {exc}); logos will be blank")
        return {}


def build_news():
    try:
        resp = requests.get("https://www.espn.com/espn/rss/nfl/news", timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        print(f"news fetch failed ({type(exc).__name__}: {exc}); falling back to previous build's headlines")
        return previous_news()

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

    team_stats = build_team_stats()
    print(f"team stats rows: {len(team_stats)}")

    team_logos = build_team_logos()
    print(f"team logos: {len(team_logos)}")

    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"build time: {build_time}")

    with open(TEMPLATE_PATH) as f:
        html = f.read()

    payloads = {
        "__BOARD_DATA__": json.dumps(rows, separators=(",", ":"), allow_nan=False),
        "__LINEUPS_DATA__": json.dumps({"asOf": lineups_asof, "teams": lineups}, separators=(",", ":"), allow_nan=False),
        "__SCHEDULE_DATA__": json.dumps({"games": games, "byes": byes}, separators=(",", ":"), allow_nan=False),
        "__NEWS_DATA__": json.dumps(news, separators=(",", ":"), allow_nan=False),
        "__TEAMSTATS_DATA__": json.dumps(team_stats, separators=(",", ":"), allow_nan=False),
        "__TEAMLOGOS_DATA__": json.dumps(team_logos, separators=(",", ":"), allow_nan=False),
        "__BUILD_TIME__": build_time,
    }
    for tag, payload in payloads.items():
        assert html.count(tag) == 1, f"expected exactly one {tag} in template"
        html = html.replace(tag, payload)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUTPUT_PATH}, {len(html)} bytes")


if __name__ == "__main__":
    main()
