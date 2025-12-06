import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from pyvis.network import Network

DATA_PATH = Path("output/board_members_1394_1404_from_urls.csv")
YEARS = range(1394, 1405)  # inclusive 1394..1404

# Palette for years (repeat if needed)
YEAR_COLORS = [
    "#e4572e","#17bebb","#ffc914","#2e86ab","#a23b72",
    "#4f5d75","#ef476f","#06d6a0","#118ab2","#073b4c","#9c27b0"
]

dtype_map = {
    "member_id": "string",
    "national_id": "string",
    "degree": "string",
    "major": "string",
    "experience": "string",
    "verification_status": "string",
    "ceo_degree": "string",
    "ceo_major": "string",
}
df = pd.read_csv(DATA_PATH, dtype=dtype_map, low_memory=False)

# Basic cleaning / keys
df = df[pd.to_numeric(df["year"], errors="coerce").between(1394, 1404)]
df["company_clean"] = df["company"].str.strip()
df["assembly_date_clean"] = df["assembly_date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
df["period_month"] = df["assembly_date_clean"].str[:6]  # YYYYMM
df["period_quarter"] = (
    df["assembly_date_clean"].str[:4]
    + "Q"
    + ((df["assembly_date_clean"].str[4:6].astype(int) - 1) // 3 + 1).astype(str)
)

def digits_only(val: str) -> str:
    return "".join(ch for ch in str(val) if ch.isdigit()) if pd.notna(val) else ""

def is_person_id(val: str) -> bool:
    """Detect person IDs even if the CSV stored them as floats (strip dots, keep digits)."""
    s = digits_only(val)
    return len(s) == 10

def clean_str(val: str) -> str:
    return "" if pd.isna(val) else str(val).strip()

def extract_representative(row):
    member_id = clean_str(row["member_id"])
    national_id = clean_str(row.get("national_id"))
    member_digits = digits_only(member_id)
    national_digits = digits_only(national_id)
    new_rep = clean_str(row.get("new_representative"))
    new_mem = clean_str(row.get("new_member"))

    person_by_member = is_person_id(member_digits)
    person_by_national = is_person_id(national_digits)
    person = person_by_member or person_by_national
    rep_name = None

    # If we have a representative string and we detect a person, use it.
    if person and new_rep and new_rep != "فاقد نماینده":
        rep_name = new_rep
    # Otherwise, if this is a person (member_id/national_id) and representative absent, take new_member as the person.
    elif person:
        rep_name = new_mem if new_mem else None

    # Build identifiers only for persons (prefer member_id digits, then national_id digits)
    rep_id = None
    if person_by_member:
        rep_id = member_digits
    elif person_by_national:
        rep_id = national_digits

    rep_key = rep_id or rep_name  # grouping key for person

    return pd.Series({"rep_name": rep_name, "rep_id": rep_id, "rep_key": rep_key, "rep_is_person": bool(rep_name)})

df[["rep_name", "rep_id", "rep_key", "rep_is_person"]] = df.apply(extract_representative, axis=1)
# Deduplicate person-company-date so multiple rows for the same assembly don't inflate edges
df = df.drop_duplicates(subset=["company_clean", "rep_key", "assembly_date_clean"])

def edges_for_year(block):
    edges = []
    people = block[block["rep_is_person"] & block["rep_key"].notna()]
    for rep_key, g in people.groupby("rep_key"):
        companies = sorted(g["company_clean"].dropna().unique())
        if len(companies) < 2:
            continue  # need at least two companies to form an edge
        rep_name = g["rep_name"].dropna().iloc[0] if g["rep_name"].notna().any() else "Unknown"
        rep_id = g["rep_id"].dropna().iloc[0] if g["rep_id"].notna().any() else rep_name
        for a, b in combinations(companies, 2):
            edges.append((a, b, rep_name, rep_id))
    return edges

def build_net(nodes, edges, year=None, color=None):
    net = Network(height="800px", width="100%", bgcolor="#0b1021", font_color="#e6ecff", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-32000, central_gravity=0.2, spring_length=140, spring_strength=0.03, damping=0.92)
    for c in nodes:
        net.add_node(c, label=c, color="#7ec8e3", shadow=True)
    for a, b, rep_name, rep_id in edges:
        title = f"Year: {year if year else 'multi'} | Rep: {rep_name} | ID: {rep_id}"
        net.add_edge(a, b, title=title, label=rep_name, color=color or "#f2d492", width=1.5, smooth=False)
    return net

def save_utf8(net, path, notebook=False):
    """Write the PyVis HTML with UTF-8 to avoid Windows cp1252 encode errors."""
    html = net.generate_html(notebook=notebook)
    Path(path).write_text(html, encoding="utf-8")

out_dir = Path("output")

def build_period_network(df_block, label, color):
    nodes = df_block["company_clean"].dropna().unique()
    edges = edges_for_year(df_block)
    if not edges:
        return None
    net = build_net(nodes, edges, year=label, color=color)
    return net, edges

# Per-year snapshots
for year in YEARS:
    block = df[df["year"] == year]
    color = YEAR_COLORS[(year - YEARS.start) % len(YEAR_COLORS)]
    res = build_period_network(block, label=year, color=color)
    if res:
        net, _ = res
        save_utf8(net, out_dir / f"board_network_{year}.html")

# Per-quarter snapshots (YYYYQ#)
quarters = sorted(df["period_quarter"].dropna().unique())
edges_by_quarter = {}
for q in quarters:
    year = int(q[:4])
    color = YEAR_COLORS[(year - YEARS.start) % len(YEAR_COLORS)]
    block = df[df["period_quarter"] == q]
    res = build_period_network(block, label=q, color=color)
    if res:
        net, _ = res
        save_utf8(net, out_dir / f"board_network_{q}.html")
        edges_by_quarter[q] = [
            {
                "from": a,
                "to": b,
                "label": rep_name,
                "title": f"Quarter: {q} | Rep: {rep_name} | ID: {rep_id}",
                "color": color,
                "width": 1.2,
                "smooth": False,
            }
            for (a, b, rep_name, rep_id) in edges_for_year(block)
        ]

# Per-month snapshots (YYYYMM)
months = sorted(df["period_month"].dropna().unique())
for m in months:
    year = int(m[:4])
    color = YEAR_COLORS[(year - YEARS.start) % len(YEAR_COLORS)]
    block = df[df["period_month"] == m]
    res = build_period_network(block, label=m, color=color)
    if res:
        net, _ = res
        save_utf8(net, out_dir / f"board_network_{m}.html")

# All-years combined (color by year)
all_nodes = df["company_clean"].dropna().unique()
all_edges = []
for year in YEARS:
    block = df[df["year"] == year]
    color = YEAR_COLORS[(year - YEARS.start) % len(YEAR_COLORS)]
    for edge in edges_for_year(block):
        all_edges.append((year, color, edge))
net_all = Network(height="850px", width="100%", bgcolor="#0b1021", font_color="#e6ecff", notebook=False, cdn_resources="in_line")
net_all.barnes_hut(gravity=-32000, central_gravity=0.2, spring_length=140, spring_strength=0.03, damping=0.92)
for c in all_nodes:
    net_all.add_node(c, label=c, color="#7ec8e3", shadow=True)
for year, color, (a, b, rep_name, rep_id) in all_edges:
    net_all.add_edge(a, b, title=f"Year: {year} | Rep: {rep_name} | ID: {rep_id}", label=rep_name, color=color, width=1.5, smooth=False)
save_utf8(net_all, out_dir / "board_network_all_years.html")

# Combined quarter network (color by year of quarter)
all_edges_q = []
for q in quarters:
    year = int(q[:4])
    color = YEAR_COLORS[(year - YEARS.start) % len(YEAR_COLORS)]
    block = df[df["period_quarter"] == q]
    for edge in edges_for_year(block):
        all_edges_q.append((q, color, edge))
net_quarters = Network(height="850px", width="100%", bgcolor="#0b1021", font_color="#e6ecff", notebook=False, cdn_resources="in_line")
net_quarters.barnes_hut(gravity=-32000, central_gravity=0.2, spring_length=140, spring_strength=0.03, damping=0.92)
for c in all_nodes:
    net_quarters.add_node(c, label=c, color="#7ec8e3", shadow=True)
for q, color, (a, b, rep_name, rep_id) in all_edges_q:
    net_quarters.add_edge(a, b, title=f"Quarter: {q} | Rep: {rep_name} | ID: {rep_id}", label=rep_name, color=color, width=1.2, smooth=False)
save_utf8(net_quarters, out_dir / "board_network_all_quarters.html")

# Combined month network (color by year of month)
all_edges_m = []
for m in months:
    year = int(m[:4])
    color = YEAR_COLORS[(year - YEARS.start) % len(YEAR_COLORS)]
    block = df[df["period_month"] == m]
    for edge in edges_for_year(block):
        all_edges_m.append((m, color, edge))
net_months = Network(height="850px", width="100%", bgcolor="#0b1021", font_color="#e6ecff", notebook=False, cdn_resources="in_line")
net_months.barnes_hut(gravity=-32000, central_gravity=0.2, spring_length=140, spring_strength=0.03, damping=0.92)
for c in all_nodes:
    net_months.add_node(c, label=c, color="#7ec8e3", shadow=True)
for m, color, (a, b, rep_name, rep_id) in all_edges_m:
    net_months.add_edge(a, b, title=f"Month: {m} | Rep: {rep_name} | ID: {rep_id}", label=rep_name, color=color, width=1.0, smooth=False)
save_utf8(net_months, out_dir / "board_network_all_months.html")

# Quarter timeline HTML with slider/play to toggle edges by quarter
def build_quarter_timeline(out_path: Path):
    if not edges_by_quarter:
        return
    quarters_list = sorted(edges_by_quarter.keys())
    initial_q = quarters_list[-1]

    net = Network(height="850px", width="100%", bgcolor="#0b1021", font_color="#e6ecff", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-32000, central_gravity=0.2, spring_length=140, spring_strength=0.03, damping=0.92)
    for c in all_nodes:
        net.add_node(c, label=c, color="#7ec8e3", shadow=True)
    # add initial quarter edges so PyVis renders correctly; later JS swaps edges
    for e in edges_by_quarter[initial_q]:
        net.add_edge(e["from"], e["to"], title=e["title"], label=e["label"], color=e["color"], width=e["width"], smooth=False)

    html = net.generate_html(notebook=False)

    controls = f"""
<div id="quarter-controls" style="padding:10px;background:#0f152d;color:#e6ecff;font-family:Arial, sans-serif;display:flex;align-items:center;gap:12px;">
  <span style="font-weight:bold;">Quarter timeline:</span>
  <input type="range" id="quarter-slider" min="0" max="{len(quarters_list)-1}" value="{len(quarters_list)-1}" step="1" style="width:260px;">
  <span id="quarter-label" style="font-weight:bold;">{initial_q}</span>
  <button id="quarter-play" style="padding:6px 10px;border:1px solid #3b4b87;background:#172041;color:#e6ecff;border-radius:6px;cursor:pointer;">Play</button>
</div>
"""
    html = html.replace('<div id="mynetwork"></div>', controls + '\n<div id="mynetwork"></div>')

    edges_json = json.dumps(edges_by_quarter)
    quarters_json = json.dumps(quarters_list)

    script = f"""
<script type="text/javascript">
  const quarters = {quarters_json};
  const edgesByQuarter = {edges_json};
  const slider = document.getElementById('quarter-slider');
  const label = document.getElementById('quarter-label');
  const playBtn = document.getElementById('quarter-play');
  let playTimer = null;

  function setQuarterByIndex(idx) {{
    const q = quarters[idx];
    if (!q) return;
    edges.clear();
    edges.add(edgesByQuarter[q] || []);
    label.textContent = q;
  }}

  slider.addEventListener('input', (e) => {{
    setQuarterByIndex(parseInt(e.target.value, 10));
  }});

  function play() {{
    if (playTimer) return;
    playTimer = setInterval(() => {{
      let idx = parseInt(slider.value, 10) + 1;
      if (idx >= quarters.length) idx = 0;
      slider.value = idx;
      setQuarterByIndex(idx);
    }}, 1200);
    playBtn.textContent = 'Pause';
  }}

  function pause() {{
    if (!playTimer) return;
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = 'Play';
  }}

  playBtn.addEventListener('click', () => {{
    if (playTimer) {{
      pause();
    }} else {{
      play();
    }}
  }});

  // lock positions after first stabilization to keep layout stable while edges change
  network.once('stabilized', function() {{
    network.stopSimulation();
  }});

  setQuarterByIndex(quarters.length - 1);
</script>
"""
    html = html.replace("</body>", script + "</body>")
    Path(out_path).write_text(html, encoding="utf-8")

build_quarter_timeline(out_dir / "board_network_timeline_quarters.html")
print("Done. Open the HTML files in your browser.")
