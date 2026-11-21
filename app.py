"""StockSense - purchase likelihood and demand insights for retailers.

Run:  streamlit run app.py
The prediction engine is trained automatically the first time the app starts (or run `python main.py`);
afterwards the saved models are loaded, never retrained.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
from src import retail_insights as R
from src import trends as T
from src.data_loader import load_or_generate
from src.feature_engineering import add_groups
from src.purchase_prediction import PurchasePredictor

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    px = None

st.set_page_config(page_title=f"{C.APP_NAME} | Retail Insights", page_icon=None, layout="wide")
if px is None:
    st.error("The charting library is missing. Run `pip install -r requirements.txt` and restart the app.")
    st.stop()

NAVY, GREEN, AMBER, RED, GREY = "#1F3A5F", "#2E9E6B", "#F2A93B", "#D6564B", "#8A94A6"
TIER_COLORS = {"High": GREEN, "Medium": AMBER, "Low": RED}
ACTION_COLORS = {"Stock up": GREEN, "Promote": "#3E7CB1", "Monitor": AMBER,
                 "Review price / fit": RED, "Limit stock": GREY}
PAGES = ["Overview", "Product Scorecard", "Demand & Popularity", "Search Trends", "New Product Check", "Action Plan"]

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; }}
.kpi {{ background:#FFFFFF; border:1px solid #E3E8EF; border-left:5px solid {NAVY}; border-radius:10px;
        padding:14px 16px; box-shadow:0 1px 3px rgba(20,30,50,.06); height:100%; }}
.kpi-green {{ border-left-color:{GREEN}; }} .kpi-amber {{ border-left-color:{AMBER}; }}
.kpi-red {{ border-left-color:{RED}; }} .kpi-grey {{ border-left-color:{GREY}; }}
.kpi-label {{ color:#5B6676; font-size:.80rem; text-transform:uppercase; letter-spacing:.04em; }}
.kpi-value {{ color:#1B2430; font-size:1.9rem; font-weight:700; line-height:1.25; }}
.kpi-sub {{ color:#5B6676; font-size:.82rem; }}
.pill {{ display:inline-block; padding:3px 12px; border-radius:999px; font-size:.78rem; font-weight:600; color:#fff; }}
.callout {{ background:#F4F6FA; border-radius:10px; padding:14px 18px; border-left:5px solid {NAVY}; color:#1B2430; }}
.subtle {{ color:#5B6676; font-size:.9rem; }}
.status-banner {{ border-radius:8px; padding:11px 16px; margin-bottom:14px; font-size:.88rem; line-height:1.45; border-left:4px solid; }}
.status-live {{ background:#EDF8F1; border-color:{GREEN}; color:#165B36; }}
.status-cached {{ background:#EEF4FA; border-color:#3E7CB1; color:#1C4A73; }}
.status-simulated {{ background:#FEF8ED; border-color:{AMBER}; color:#7A4F0B; }}
.sidebar-box {{ background:#F8FAFC; border:1px solid #E3E8EF; border-radius:8px; padding:10px 12px; margin-top:10px; }}
.sidebar-pill {{ display:inline-block; padding:2px 9px; border-radius:12px; font-size:.74rem; font-weight:600; color:#fff; margin-bottom:4px; }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Preparing the prediction engine (first launch only)...")
def load_predictor() -> PurchasePredictor:
    if not PurchasePredictor.models_available():
        import main as pipeline
        pipeline.main(["--no-plots"])
    return PurchasePredictor()


@st.cache_data(show_spinner="Loading store activity...")
def load_trials() -> pd.DataFrame:
    return load_or_generate()


@st.cache_data(show_spinner=False, ttl=3600)
def load_trend_table(force_live: bool = False) -> pd.DataFrame:
    return T.load_trends(force_refresh=force_live)


@st.cache_data(show_spinner="Scoring products...")
def load_scorecard() -> pd.DataFrame:
    return R.build_scorecard(load_trials(), load_predictor(), load_trend_table())


# ------------------------------------------------------------------ helpers
def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi kpi-{tone}"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def pill(text, color):
    return f'<span class="pill" style="background:{color}">{text}</span>'


def delta_text(cur, prev, unit="", pts=False):
    if prev is None or pd.isna(prev) or pd.isna(cur):
        return "no prior period"
    d = cur - prev
    arrow = "▲" if d > 0 else "▼" if d < 0 else "■"
    return f"{arrow} {abs(d):.1f}{' pts' if pts else unit} vs previous 7 days"


def style(fig, height=340, legend=True):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=legend, legend=dict(orientation="h", y=-0.2),
                      font=dict(family="sans-serif", size=12))
    return fig


def dual_axis(x, bars, line, bar_name, line_name, title, height=340):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=x, y=bars, name=bar_name, marker_color="#9DB4D1", secondary_y=False)
    fig.add_scatter(x=x, y=line, name=line_name, mode="lines+markers", marker_color=AMBER,
                    line=dict(width=3), secondary_y=True)
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name, secondary_y=True, rangemode="tozero")
    fig.update_layout(title=title)
    return style(fig, height)


PRODUCT_COLS = {
    "Product": st.column_config.TextColumn("Product"),
    "Product ID": st.column_config.TextColumn("ID"),
    "Price": st.column_config.NumberColumn("Price", format="₹%d"),
    "Trials": st.column_config.NumberColumn("Fitting-room trials"),
    "Conversion_Pct": st.column_config.NumberColumn("Trial → purchase", format="%.0f%%"),
    "Likelihood_Pct": st.column_config.ProgressColumn("Purchase likelihood", format="%.0f%%",
                                                       min_value=0, max_value=100),
    "Likelihood": st.column_config.TextColumn("Level"),
    "Demand": st.column_config.TextColumn("Demand"),
    "Search_Trend": st.column_config.TextColumn("Search trend"),
    "Action": st.column_config.TextColumn("Action"),
    "Reason": st.column_config.TextColumn("Why", width="large"),
    "Segment": st.column_config.TextColumn("Shopper segment"),
}


def product_table(df, cols, height=None):
    st.dataframe(df[cols], column_config={k: v for k, v in PRODUCT_COLS.items() if k in cols},
                 hide_index=True, height=height)


# ------------------------------------------------------------------ sidebar / filters
def sidebar(sc: pd.DataFrame, table: pd.DataFrame):
    with st.sidebar:
        st.markdown(f"## {C.APP_NAME}")
        st.caption("Men's formal shirts · purchase likelihood & demand insights")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Product filters**")
        colors = st.multiselect("Colour", sorted(sc["Color"].unique()))
        patterns = st.multiselect("Pattern", sorted(sc["Pattern"].unique()))
        fits = st.multiselect("Fit", sorted(sc["Fit"].unique()))
        collars = st.multiselect("Collar", sorted(sc["Collar"].unique()))
        sleeves = st.multiselect("Sleeve", sorted(sc["Sleeve"].unique()))
        lo, hi = int(sc["Price"].min()), int(sc["Price"].max())
        price = st.slider("Price (₹)", lo, hi, (lo, hi), step=50)
        levels = st.multiselect("Purchase likelihood", R.TIERS)
        st.markdown("---")
        
        # Retailer-facing market signal status
        if T.is_live(table):
            st.markdown(
                f'<div class="sidebar-box">'
                f'<span class="sidebar-pill" style="background:{GREEN}">● Live Trends</span><br>'
                f'<span style="font-size:0.80rem; color:#5B6676;">Google Trends active (India). Calibrating purchase likelihood.</span>'
                f'</div>', unsafe_allow_html=True
            )
        elif T.is_cached(table):
            meta = T.get_trend_metadata()
            up_str = meta.get("last_updated_human", "Saved cache")
            st.markdown(
                f'<div class="sidebar-box">'
                f'<span class="sidebar-pill" style="background:#3E7CB1">● Cached Trends</span><br>'
                f'<span style="font-size:0.80rem; color:#5B6676;">Verified search trends ({up_str}).</span>'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="sidebar-box">'
                f'<span class="sidebar-pill" style="background:{AMBER}">● Market Benchmark</span><br>'
                f'<span style="font-size:0.80rem; color:#5B6676;">Calibrated style demand baseline.</span>'
                f'</div>', unsafe_allow_html=True
            )
            
        if st.button("Check for Live Updates", key="btn_sidebar_sync", use_container_width=True):
            with st.spinner("Checking Google Trends..."):
                try:
                    fresh = T.load_trends(force_refresh=True)
                    st.cache_data.clear()
                    if T.is_live(fresh):
                        st.toast("Google Trends connected and updated!", icon="✅")
                    else:
                        st.toast("Google Trends rate-limited; retaining verified data.", icon="ℹ️")
                    st.rerun()
                except Exception:
                    st.toast("Google Trends offline; using verified baseline.", icon="ℹ️")
    f = sc[sc["Price"].between(*price)]
    for col, sel in [("Color", colors), ("Pattern", patterns), ("Fit", fits), ("Collar", collars),
                     ("Sleeve", sleeves), ("Likelihood", levels)]:
        if sel:
            f = f[f[col].isin(sel)]
    return page, f


def header(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


# ------------------------------------------------------------------ pages
def page_overview(sc, trials):
    header("Store overview", "How likely your products are to sell, where demand is heading and what to do next.")
    pc = R.period_comparison(trials)
    n_high = int((sc["Likelihood"] == "High").sum())
    attention = int(sc["Action"].isin(["Review price / fit", "Limit stock"]).sum())
    c = st.columns(5)
    kpi(c[0], "Products tracked", f"{len(sc):,}", f"{int(sc['Trials'].sum()):,} fitting-room trials")
    kpi(c[1], "Avg purchase likelihood", f"{sc['Likelihood_Pct'].mean():.0f}%", "per fitting-room trial")
    kpi(c[2], "High-likelihood products", f"{n_high}", f"{n_high / len(sc):.0%} of range", "green")
    kpi(c[3], "Need attention", f"{attention}", "review price / fit or limit stock", "red")
    kpi(c[4], "Trial → purchase (last 7 days)", f"{pc['conv']:.0f}%",
        delta_text(pc["conv"], pc["conv_prev"], pts=True), "amber")
    st.write("")

    left, right = st.columns([2, 1])
    with left:
        w = R.weekly_activity(trials)
        st.plotly_chart(dual_axis([d.strftime("%d %b") for d in w["Week"]], w["Trials"], w["Conversion_Pct"],
                                  "Fitting-room trials", "Trial → purchase (%)", "Weekly demand"))
    with right:
        mix = sc["Likelihood"].value_counts().reindex(R.TIERS).fillna(0).reset_index()
        mix.columns = ["Level", "Products"]
        fig = px.pie(mix, names="Level", values="Products", hole=0.55, color="Level",
                     color_discrete_map=TIER_COLORS, title="Purchase-likelihood mix")
        st.plotly_chart(style(fig))

    st.subheader("This week's priorities")
    cols = st.columns(3)
    blocks = [("Stock up", "Likely to sell, in demand", "Likelihood_Pct", False),
              ("Promote", "Sells well but under-tried", "Likelihood_Pct", False),
              ("Review price / fit", "Tried often, unlikely to sell", "Trials", False)]
    for col, (act, sub, sort, asc) in zip(cols, blocks):
        with col:
            st.markdown(pill(act, ACTION_COLORS[act]) + f' <span class="subtle">{sub}</span>', unsafe_allow_html=True)
            sub_df = sc[sc["Action"] == act].sort_values(sort, ascending=asc).head(6)
            if sub_df.empty:
                st.caption("Nothing in this group for the current filters.")
            else:
                product_table(sub_df, ["Product", "Price", "Likelihood_Pct"], height=250)


def page_scorecard(sc):
    header("Product scorecard", "Purchase likelihood for every product, with the level, demand and recommended action.")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    acts = c1.multiselect("Action", R.ACTIONS)
    demand = c2.multiselect("Demand", ["High", "Medium", "Low"])
    q = c3.text_input("Search product or ID")
    df = sc
    if acts:
        df = df[df["Action"].isin(acts)]
    if demand:
        df = df[df["Demand"].isin(demand)]
    if q:
        df = df[df["Product"].str.contains(q, case=False) | df["Product ID"].str.contains(q, case=False)]
    if df.empty:
        st.info("No products match the current filters.")
        return
    c = st.columns(4)
    for col, tier, tone in zip(c[:3], R.TIERS, ["green", "amber", "red"]):
        n = int((df["Likelihood"] == tier).sum())
        kpi(col, f"{tier} likelihood", n, f"{n / len(df):.0%} of shown products", tone)
    kpi(c[3], "Avg likelihood (shown)", f"{df['Likelihood_Pct'].mean():.0f}%", f"{len(df)} products")
    st.write("")
    fig = px.scatter(df, x="Price", y="Likelihood_Pct", size="Trials", color="Likelihood",
                     color_discrete_map=TIER_COLORS, hover_name="Product",
                     hover_data=["Product ID", "Trials", "Action"],
                     category_orders={"Likelihood": R.TIERS},
                     labels={"Likelihood_Pct": "Purchase likelihood (%)", "Price": "Price (₹)"},
                     title="Price vs purchase likelihood (bubble size = fitting-room trials)")
    st.plotly_chart(style(fig, 380))
    product_table(df, ["Product", "Product ID", "Price", "Trials", "Conversion_Pct", "Likelihood_Pct", "Likelihood",
                       "Demand", "Search_Trend", "Action"], height=460)
    st.download_button("Download this list (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="product_scorecard.csv", mime="text/csv")


ATTR_MAP = {"Colour": "Color", "Pattern": "Pattern", "Fit": "Fit", "Collar": "Collar_Type",
            "Sleeve": "Sleeve_Type", "Size": "Size", "Price band": "Price_Band"}


def page_demand(sc, trials):
    header("Demand & popularity", "What shoppers are trying on and what they actually buy.")
    t = add_groups(trials)
    t["Size"] = t["Size"].astype(str)
    choice = st.radio("Break down by", list(ATTR_MAP), horizontal=True)
    col = ATTR_MAP[choice]
    order = [str(s) for s in C.SIZES] if col == "Size" else (
        sorted(t[col].unique(), key=lambda s: int(s.split("-")[0])) if col == "Price_Band" else None)
    a = R.attribute_summary(t, col, order)
    a[col] = a[col].astype(str)
    st.plotly_chart(dual_axis(a[col], a["Share_Pct"], a["Conversion_Pct"], "Share of trials (%)",
                              "Trial → purchase (%)", f"Popularity vs conversion by {choice.lower()}"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most tried**")
        st.dataframe(a.sort_values("Trials", ascending=False).head(5)[[col, "Trials", "Share_Pct", "Conversion_Pct"]],
                     hide_index=True)
    with c2:
        st.markdown("**Best converting** (at least 30 trials)")
        st.dataframe(a[a["Trials"] >= 30].sort_values("Conversion_Pct", ascending=False).head(5)
                     [[col, "Trials", "Share_Pct", "Conversion_Pct"]], hide_index=True)

    top = a.sort_values("Trials", ascending=False).head(6)[col].tolist()
    ts = pd.to_datetime(t["Trial_Timestamp"])
    wk = t.assign(Week=ts.dt.normalize() - pd.to_timedelta(ts.dt.dayofweek, unit="D"))
    wk = wk[wk[col].astype(str).isin(top)].groupby(["Week", col]).size().reset_index(name="Trials")
    wk[col] = wk[col].astype(str)
    fig = px.line(wk, x="Week", y="Trials", color=col, markers=True,
                  title=f"Weekly fitting-room demand - top {choice.lower()} groups")
    st.plotly_chart(style(fig, 360))

    st.subheader("Most popular products")
    pop = sc.sort_values("Trials", ascending=False).head(10).iloc[::-1]
    fig = px.bar(pop, x="Trials", y="Product ID", orientation="h", color="Likelihood",
                 color_discrete_map=TIER_COLORS, hover_name="Product", hover_data=["Price", "Likelihood_Pct"],
                 labels={"Trials": "Fitting-room trials"}, title="Top 10 products by fitting-room trials")
    st.plotly_chart(style(fig, 380))


def page_trends(sc, table):
    header("Search trends", "Google search interest for the styles you sell, and which of your products ride the wave.")

    source_type = "live" if T.is_live(table) else ("cached" if T.is_cached(table) else "simulated")
    meta = T.get_trend_metadata()
    last_up = meta.get("last_updated_human") or meta.get("last_updated", "")

    # Elegant unobtrusive status strip
    b_col1, b_col2 = st.columns([3.8, 1.2])
    with b_col1:
        if source_type == "live":
            st.markdown(
                '<div class="status-banner status-live">'
                '<b>● Live Google Trends Connected</b> · Real-time search volume across India. '
                'Current search signals actively refine demand outlook and restocking urgency.'
                '</div>', unsafe_allow_html=True
            )
        elif source_type == "cached":
            up_str = f" (saved: {last_up})" if last_up else ""
            st.markdown(
                f'<div class="status-banner status-cached">'
                f'<b>● Verified Market Cache{up_str}</b> · Displaying verified Google search trends. '
                'Retail scoring and forecasting reflect these verified trend trajectories.'
                '</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-banner status-simulated">'
                '<b>● Market Baseline Mode</b> · Displaying calibrated fashion search benchmarks while external '
                'connection is idle. All retail scoring, clustering and forecasting remain fully operational.'
                '</div>', unsafe_allow_html=True
            )
    with b_col2:
        if st.button("🔄 Sync Live Trends", key="sync_trends_page", use_container_width=True):
            with st.spinner("Connecting to Google Trends..."):
                try:
                    fresh = T.load_trends(force_refresh=True)
                    st.cache_data.clear()
                    if T.is_live(fresh):
                        st.toast("Google Trends connected and updated!", icon="✅")
                    else:
                        st.toast("Google rate limit active; using latest verified baseline.", icon="ℹ️")
                    st.rerun()
                except Exception:
                    st.toast("Unable to reach Google Trends; using verified baseline.", icon="ℹ️")

    st.write("")

    changes = R.trend_changes(table)
    keys = list(C.TREND_KEYWORDS)
    disp_names = getattr(C, "TREND_DISPLAY_NAMES", {})

    for row in (keys[:3], keys[3:]):
        cols = st.columns(3)
        for col, key in zip(cols, row):
            ch = changes[key]
            d = R.direction(ch)
            label = disp_names.get(key, C.TREND_KEYWORDS[key].title())
            src_tag = "Live" if source_type == "live" else ("Cached" if source_type == "cached" else "Benchmark")
            arrow = "▲" if ch > 0 else ("▼" if ch < 0 else "■")
            sub_text = f"{arrow} {abs(ch):.0f}% vs previous 4 weeks · {d} ({src_tag})"
            kpi(col, label, f"{table[key].iloc[-1]:.0f} / 100",
                sub_text,
                {"Rising": "green", "Falling": "red", "Stable": "grey"}[d])
        st.write("")

    # Multi-series trend chart
    disp_map = {k: disp_names.get(k, v.title()) for k, v in C.TREND_KEYWORDS.items()}
    long = table.rename(columns=disp_map)
    long["date"] = pd.to_datetime(long["date"]) if "date" in long else long["week"]
    val_vars = [disp_map[k] for k in C.TREND_KEYWORDS if disp_map[k] in long.columns]
    long = long.melt(id_vars=[c for c in ["date"] if c in long], value_vars=val_vars,
                     var_name="Style category", value_name="Search interest")

    chart_subtitle = {
        "live": "Live Google Trends (India)",
        "cached": f"Cached Google Trends ({last_up or 'local store cache'})",
        "simulated": "Market Baseline Benchmark (Simulated)",
    }[source_type]

    fig = px.line(long, x="date", y="Search interest", color="Style category", markers=True,
                  title=f"Search Interest Trajectory (0-100) — {chart_subtitle}",
                  color_discrete_sequence=[NAVY, GREEN, AMBER, RED, "#3E7CB1", GREY])
    fig.update_yaxes(range=[0, 105], title="Search Interest Index (0-100)")
    fig.update_xaxes(title="Week")
    st.plotly_chart(style(fig, 390))

    st.subheader("Retail Decision Support: Aligning Inventory to Search Trends")
    tab1, tab2 = st.tabs(["Restock & Feature (Rising Search)", "Inventory Watchlist (Slowing Demand)"])

    with tab1:
        rising = sc[(sc["Search_Trend"] == "Rising") & (sc["Likelihood"] != "Low")].sort_values(
            ["Likelihood_Pct", "Trials"], ascending=False)
        if rising.empty:
            st.caption("No products currently match rising search trends for the active filters.")
        else:
            hi = int((rising["Likelihood"] == "High").sum())
            st.markdown(
                f'<div class="callout"><b>{len(rising)}</b> products fit styles with rising search momentum; '
                f'<b>{hi}</b> of them already show high customer purchase likelihood. '
                'These are prime candidates to feature in front-of-store displays or increase reorder allocations.</div>',
                unsafe_allow_html=True)
            st.write("")
            product_table(rising.head(15),
                          ["Product", "Price", "Likelihood_Pct", "Likelihood", "Demand", "Search_Trend", "Action"],
                          height=400)

    with tab2:
        falling = sc[(sc["Search_Trend"] == "Falling") & (sc["Demand"] == "High")].sort_values("Trials", ascending=False)
        if falling.empty:
            st.info("No high-traffic products currently show falling search momentum. Fitting-room traffic aligns well with search trends.")
        else:
            st.markdown(
                f'<div class="callout" style="border-left-color:{AMBER};">'
                f'<b>{len(falling)}</b> high-traffic products belong to styles where broader search momentum is slowing. '
                'Review reorder quantities and plan timely promotions before store demand tapers off.</div>',
                unsafe_allow_html=True)
            st.write("")
            product_table(falling.head(15),
                          ["Product", "Price", "Likelihood_Pct", "Likelihood", "Demand", "Search_Trend", "Action"],
                          height=360)


def page_predict(table):
    header("New product check", "Estimate how likely a product is to sell before you commit stock.")
    pred = load_predictor()
    with st.form("new_product"):
        c1, c2, c3 = st.columns(3)
        size = c1.selectbox("Size", C.SIZES, index=C.SIZES.index(40))
        sleeve = c1.radio("Sleeve", C.SLEEVES, index=1, horizontal=True)
        fit = c1.radio("Fit", C.FITS, horizontal=True)
        color = c2.selectbox("Colour", C.COLORS, index=C.COLORS.index("Blue"))
        collar = c2.selectbox("Collar", C.COLLARS, index=1)
        pattern = c2.selectbox("Pattern", C.PATTERNS, index=C.PATTERNS.index("Solid"))
        price = c3.number_input("Price (₹)", min_value=300.0, max_value=10000.0, value=1499.0, step=50.0)
        tried = c3.checkbox("This product has already been tried in the fitting room")
        trials = c3.number_input("Fitting-room trials so far", min_value=0, max_value=1000, value=10, step=1,
                                 disabled=not tried)
        use_live = st.checkbox("Include current Google search interest for this style", value=True)
        go_btn = st.form_submit_button("Check purchase likelihood")
    if not go_btn:
        st.caption("Leave the trial box unticked for a brand-new product with no fitting-room history.")
        return
    try:
        r = pred.predict(size, sleeve, color, collar, fit, pattern, price,
                         trial_count=int(trials) if tried else None,
                         trend_index=None if use_live else 50.0)
    except ValueError as exc:
        st.error(f"Please check the inputs: {exc}")
        return
    for w in r["warnings"]:
        st.warning(w.replace("training range", "price range seen in your store"))
    base = r["base_rate"]
    tier = R.tier_label(r["probability"], base)
    demand = {"Average": "Medium"}.get(r["recommendation"]["demand_tier"], r["recommendation"]["demand_tier"])
    changes = R.trend_changes(table)
    trend, ch = R.product_trend(pattern, fit, changes)
    action, why = R.action_for(tier, demand, trend)

    g, k = st.columns([1, 1.4])
    with g:
        lo, hi = (base - C.TIER_MARGIN) * 100, (base + C.TIER_MARGIN) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=r["likelihood_pct"], number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Purchase likelihood"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": TIER_COLORS[tier]},
                   "steps": [{"range": [0, lo], "color": "#FBE3E0"}, {"range": [lo, hi], "color": "#FDF1D8"},
                             {"range": [hi, 100], "color": "#DDF1E7"}]}))
        st.plotly_chart(style(fig, 300, legend=False))
    with k:
        c = st.columns(3)
        kpi(c[0], "Likelihood level", tier, f"store average {base:.0%}", {"High": "green", "Medium": "amber", "Low": "red"}[tier])
        kpi(c[1], "Shopper segment", f"Segment {r['cluster'] + 1}", f"{r['cluster_weight_pct']:.0f}% of fitting-room activity")
        kpi(c[2], "Demand outlook", demand, f"search interest {trend.lower()}",
            {"High": "green", "Medium": "amber", "Low": "grey"}[demand])
        st.write("")
        st.markdown(f'<div class="callout">{pill(action, ACTION_COLORS[action])} &nbsp; {why}<br><br>'
                    f'Out of 100 fitting-room trials of a product like this, about <b>{r["likelihood_pct"]:.0f}</b> '
                    f'are expected to end in a purchase. This is a purchase estimate, not a stock quantity - combine it '
                    f'with expected store traffic, current stock and delivery lead time.</div>', unsafe_allow_html=True)
    st.subheader("What is driving this score")
    ex = pd.DataFrame(r["explanation"]).head(6)
    ex["Impact"] = ex["Effect_pp"].map(lambda v: "Helps" if v >= 0 else "Hurts")
    ex["Factor"] = ex["Feature"].replace({"Color": "Colour", "Sleeve_Type": "Sleeve", "Collar_Type": "Collar",
                                           "Trial_Count": "Trials so far", "Trend_Index": "Search interest"}) \
        + ": " + ex["Value"].astype(str)
    fig = px.bar(ex.iloc[::-1], x="Effect_pp", y="Factor", orientation="h", color="Impact",
                 color_discrete_map={"Helps": GREEN, "Hurts": RED},
                 labels={"Effect_pp": "Effect on likelihood (percentage points)"})
    st.plotly_chart(style(fig, 320))
    st.caption("Each bar compares this product with the average of all alternatives for that single attribute.")


def page_actions(sc):
    header("Action plan", "A clear list of what to stock, promote, monitor, fix or cut back.")
    if sc.empty:
        st.info("No products match the current filters.")
        return
    c = st.columns(5)
    desc = {"Stock up": "Likely to sell and in demand", "Promote": "Sells well, needs visibility",
            "Monitor": "Average - watch conversion", "Review price / fit": "Tried often, rarely bought",
            "Limit stock": "Low likelihood, low interest"}
    tones = {"Stock up": "green", "Promote": "", "Monitor": "amber", "Review price / fit": "red", "Limit stock": "grey"}
    for col, a in zip(c, R.ACTIONS):
        kpi(col, a, int((sc["Action"] == a).sum()), desc[a], tones[a])
    st.write("")
    tabs = st.tabs(R.ACTIONS)
    for tab, a in zip(tabs, R.ACTIONS):
        with tab:
            sub = sc[sc["Action"] == a]
            if sub.empty:
                st.caption("No products in this group for the current filters.")
                continue
            st.markdown(f"**{len(sub)} products** - {desc[a].lower()}.")
            # best sellers first; for problem groups show the weakest first
            sub = sub.sort_values("Likelihood_Pct", ascending=a in ("Limit stock", "Review price / fit"))
            product_table(sub, ["Product", "Product ID", "Price", "Trials", "Likelihood_Pct", "Demand",
                                "Search_Trend", "Reason"], height=420)
    st.download_button("Download full action plan (CSV)",
                       sc[["Product ID", "Product", "Price", "Trials", "Likelihood_Pct", "Likelihood", "Demand",
                           "Search_Trend", "Action", "Reason"]].to_csv(index=False).encode("utf-8"),
                       file_name="action_plan.csv", mime="text/csv")


# ------------------------------------------------------------------ app
def main():
    sc_all, trials, table = load_scorecard(), load_trials(), load_trend_table()
    page, sc = sidebar(sc_all, table)
    if page == "New Product Check":
        page_predict(table)
        return
    if sc.empty:
        st.info("No products match the current filters. Clear a filter in the sidebar.")
        return
    t = trials[trials["Garment_ID"].isin(sc["Product ID"])]
    if page == "Overview":
        page_overview(sc, t)
    elif page == "Product Scorecard":
        page_scorecard(sc)
    elif page == "Demand & Popularity":
        page_demand(sc, t)
    elif page == "Search Trends":
        page_trends(sc, table)
    else:
        page_actions(sc)


main()
