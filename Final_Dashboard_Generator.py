import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import base64
import re
import time
import random
from datetime import datetime, timedelta
from google_play_scraper import search, reviews
import sys
import concurrent.futures

# --- CONFIGURATION FOR PRESENTATION ---
SEARCH_TERMS = [
    'productivity', 'game', 'social', 'finance', 'health', 'education', 
    'weather', 'shopping', 'dating', 'travel', 'music', 'news', 'sports', 
    'crypto', 'art', 'business', 'medical', 'food', 'lifestyle', 'simulation',
    'action', 'adventure', 'puzzle', 'racing', 'rpg', 'strategy', 'tools',
    'personalization', 'photography', 'video players', 'house & home'
]
HITS_PER_TERM = 30  # Reduced slightly to ensure rate limits aren't hit with deep scraping

# --- GLOBAL NLP SETUP (INITIALIZED ONCE FOR THREAD SAFETY) ---
print("INITIALIZING NLP CORE...")
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# --- CONSOLE VISUAL EFFECTS ---
def print_system_msg(msg, type="INFO"):
    colors = {"INFO": "\033[94m", "WARN": "\033[93m", "CRIT": "\033[91m", "OK": "\033[92m", "END": "\033[0m"}
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{colors.get(type, '')}[{timestamp}] {type.center(4)} ¦ {msg}{colors['END']}")

def matrix_loader():
    chars = "/—\\|" 
    for _ in range(10):
        for char in chars:
            sys.stdout.write(f'\r\033[92m[ CONNECTING TO GOOGLE PLAY API... {char} ]\033[0m')
            sys.stdout.flush()
            time.sleep(0.1)

# --- 1. NEW FEATURE: REAL SENTIMENT ANALYSIS ---
def get_real_sentiment(app_id):
    """Fetches real user reviews to calculate sentiment, bypassing developer descriptions."""
    try:
        result, _ = reviews(
            app_id,
            lang='en', 
            country='us', 
            count=20 # Analyze last 20 reviews for speed
        )
        if not result:
            return 0.0
        # Combine all review text
        text = " ".join([str(r.get('content', '')) for r in result])
        return sia.polarity_scores(text)['compound']
    except Exception:
        return 0.0

# --- 1b. THREADED SCRAPER WORKER ---
def scrape_category(term):
    """Worker function for concurrent execution."""
    sys.stdout.write(f"\r\033[96m> THREAD SPAWNED: ANALYZING SECTOR [{term.upper()}] \033[0m")
    sys.stdout.flush()
    
    cat_apps = []
    try:
        results = search(term, lang='en', country='us', n_hits=HITS_PER_TERM)
        
        for r in results:
            try:
                installs = int(str(r.get('installs', '0')).replace(',', '').replace('+', ''))
                rating = float(r.get('score', 0) or 0)
                price = float(r.get('price', 0) or 0)
                
                # --- HYBRID SENTIMENT STRATEGY ---
                # To keep speed high, we mix description sentiment with real review sentiment
                # Real reviews are fetched only for high-install apps to optimize API calls
                if installs > 1000000:
                    sentiment = get_real_sentiment(r['appId'])
                else:
                    summary = str(r.get('summary', '')) + " " + str(r.get('description', ''))[:100]
                    sentiment = sia.polarity_scores(summary)['compound']
                
                # --- SYNTHETIC ENRICHMENT ---
                daily_active_users = int(installs * random.uniform(0.05, 0.2))
                churn_rate = random.uniform(0.02, 0.15)
                server_latency = random.randint(20, 500)
                revenue_per_user = price + random.uniform(0.1, 2.0) if price > 0 else random.uniform(0.01, 0.5)
                
                # Success Probability Calculation
                norm_install = min(installs / 1000000, 1) 
                norm_rating = rating / 5
                success_score = (norm_install * 0.4) + (norm_rating * 0.4) + ((sentiment + 1) / 2 * 0.2)
                
                cat_apps.append({
                    'App': r['title'],
                    'Category': term.capitalize(),
                    'Rating': rating,
                    'Installs': installs,
                    'Price': price,
                    'Type': 'Paid' if price > 0 else 'Free',
                    'Sentiment': sentiment,
                    'Success_Score': success_score * 100, 
                    'Daily_Active_Users': daily_active_users,
                    'Churn_Rate': churn_rate,
                    'Server_Latency_ms': server_latency,
                    'Est_Monthly_Revenue': daily_active_users * revenue_per_user * 30,
                    'Developer': r.get('developer', 'Unknown')
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Error in {term}: {e}")
        
    return cat_apps

# --- 2. HYPER-SCALED DATA ENGINE & ML PIPELINE ---
def fetch_massive_data():
    matrix_loader()
    print("\n")
    print_system_msg("INITIALIZING NEURAL SCRAPER v6.0 (MULTI-THREADED)", "OK")
    
    all_apps = []
    
    # --- NEW FEATURE: THREAD POOL EXECUTION ---
    # Scrapes all categories simultaneously instead of one by one
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Map the scrape_category function to all search terms
        results = list(executor.map(scrape_category, SEARCH_TERMS))
        
        # Flatten the list of lists
        for sublist in results:
            all_apps.extend(sublist)
            
    print("\n")
    print_system_msg(f"DATA INGESTION COMPLETE. ELAPSED: {time.time() - start_time:.2f}s", "OK")
    print_system_msg(f"PROCESSED {len(all_apps)} ENTITIES.", "OK")
    
    df = pd.DataFrame(all_apps)
    df['Rating'] = df['Rating'].fillna(0)
    df['Installs'] = df['Installs'].fillna(0)

    # --- ADVANCED AI: K-MEANS CLUSTERING ---
    print_system_msg("RUNNING UNSUPERVISED ML CLUSTERING (K-MEANS)...", "INFO")
    try:
        scaler = StandardScaler()
        features = df[['Rating', 'Installs', 'Sentiment', 'Success_Score']]
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster_ID'] = kmeans.fit_predict(scaled_features)
        
        cluster_means = df.groupby('Cluster_ID')['Installs'].mean().sort_values(ascending=False)
        sorted_ids = cluster_means.index.tolist()
        cluster_names = {}
        if len(sorted_ids) >= 4:
            cluster_names[sorted_ids[0]] = "Market Titans"     
            cluster_names[sorted_ids[1]] = "Volume Leaders"    
            cluster_names[sorted_ids[2]] = "Niche Gems"        
            cluster_names[sorted_ids[3]] = "Emerging/Risky"    
        
        df['Cluster_Label'] = df['Cluster_ID'].map(cluster_names).fillna("General")
        
    except Exception as e:
        print_system_msg(f"ML CLUSTERING FAILED: {e}", "WARN")
        df['Cluster_Label'] = "General"

    # --- KEYWORD EXTRACTION ---
    print_system_msg("EXTRACTING HIGH-VALUE KEYWORDS...", "INFO")
    all_titles = " ".join(df['App'].astype(str))
    all_titles = re.sub(r'[^a-zA-Z\s]', '', all_titles).lower()
    tokens = [word for word in all_titles.split() if word not in stop_words and len(word) > 3]
    keyword_counts = Counter(tokens).most_common(20)
    df_keywords = pd.DataFrame(keyword_counts, columns=['Keyword', 'Frequency'])
    
    # --- SYNTHETIC TIME SERIES ---
    print_system_msg("GENERATING 30-DAY HISTORICAL BACKLOG...", "INFO")
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    dates.reverse()
    
    trend_data = []
    top_cats = df['Category'].value_counts().nlargest(5).index.tolist()
    
    for date in dates:
        for cat in top_cats:
            base_install = df[df['Category'] == cat]['Installs'].mean()
            noise = random.uniform(0.9, 1.1) 
            trend_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Category': cat,
                'Avg_Installs': base_install * noise,
                'Sentiment_Trend': random.uniform(-0.5, 0.8)
            })
            
    df_trends = pd.DataFrame(trend_data)
    
    return df, df_trends, df_keywords

# --- 3. THE VISUALIZATION FACTORY ---
def create_command_center_charts(df, df_trends, df_keywords):
    print_system_msg("RENDERING HIGH-FIDELITY VECTOR GRAPHICS...", "INFO")
    charts = {}
    template = "plotly_dark"
    layout_bg = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Segoe UI, sans-serif"))

    # A. GLOBAL OVERVIEW
    fig1 = px.sunburst(df, path=['Type', 'Cluster_Label', 'Category'], values='Installs', 
                       title="<b>AI Cluster & Market Hierarchy</b>", color='Sentiment', color_continuous_scale='RdBu')
    fig1.update_layout(template=template, **layout_bg)
    charts['market_sunburst'] = fig1.to_html(full_html=False, include_plotlyjs='cdn')

    fig2 = px.treemap(df, path=['Category', 'App'], values='Est_Monthly_Revenue', 
                      title="<b>Revenue Ecosystem Map</b>", color='Success_Score', color_continuous_scale='Viridis')
    fig2.update_layout(template=template, **layout_bg)
    charts['revenue_treemap'] = fig2.to_html(full_html=False, include_plotlyjs='cdn')

    # B. ML & PREDICTION
    fig3 = px.scatter(df, x='Rating', y='Sentiment', color='Cluster_Label', size='Installs',
                      title="<b>AI Cluster Analysis (K-Means)</b>", hover_data=['App'])
    fig3.update_layout(template=template, **layout_bg)
    charts['ml_clusters'] = fig3.to_html(full_html=False, include_plotlyjs='cdn')

    fig4 = px.bar(df_keywords, x='Frequency', y='Keyword', orientation='h',
                  title="<b>Top Trending App Keywords</b>", color='Frequency', color_continuous_scale='Plasma')
    fig4.update_layout(template=template, **layout_bg, yaxis=dict(autorange="reversed"))
    charts['keywords'] = fig4.to_html(full_html=False, include_plotlyjs='cdn')

    # C. LIVE TRENDS
    fig5 = px.line(df_trends, x='Date', y='Avg_Installs', color='Category', 
                   title="<b>30-Day Growth Trajectory</b>", markers=True)
    fig5.update_layout(template=template, **layout_bg, hovermode="x unified")
    charts['trend_installs'] = fig5.to_html(full_html=False, include_plotlyjs='cdn')

    fig6 = px.area(df_trends, x='Date', y='Sentiment_Trend', color='Category',
                   title="<b>Sentiment Volatility Index</b>")
    fig6.update_layout(template=template, **layout_bg)
    charts['trend_sentiment'] = fig6.to_html(full_html=False, include_plotlyjs='cdn')

    # D. DEEP METRICS & NEW BATTLE MODE
    fig7 = px.scatter_3d(df, x='Rating', y='Price', z='Success_Score', color='Cluster_Label',
                         title="<b>Success Probability Topology (3D)</b>", opacity=0.7, size_max=5)
    fig7.update_layout(template=template, **layout_bg, scene=dict(bgcolor='rgba(0,0,0,0)'))
    charts['3d_perf'] = fig7.to_html(full_html=False, include_plotlyjs='cdn')
    
    # --- NEW FEATURE: BATTLE MODE RADAR CHART ---
    # Select top 2 apps dynamically for comparison
    try:
        top_apps = df.sort_values(by='Success_Score', ascending=False).head(2)
        app_a = top_apps.iloc[0]
        app_b = top_apps.iloc[1]
        
        # Normalize installs for visualization (log scale logic or cap)
        cap = 100000000
        inst_a = min(app_a['Installs'], cap) / cap * 10
        inst_b = min(app_b['Installs'], cap) / cap * 10
        
        fig_battle = go.Figure()
        fig_battle.add_trace(go.Scatterpolar(
              r=[app_a['Rating'], (app_a['Sentiment']+1)*2.5, inst_a, app_a['Success_Score']/20],
              theta=['Rating (5)', 'Sentiment (5)', 'Installs (Norm)', 'Score (5)'],
              fill='toself',
              name=f"{app_a['App'][:15]}..."
        ))
        fig_battle.add_trace(go.Scatterpolar(
              r=[app_b['Rating'], (app_b['Sentiment']+1)*2.5, inst_b, app_b['Success_Score']/20],
              theta=['Rating', 'Sentiment', 'Installs', 'Score'],
              fill='toself',
              name=f"{app_b['App'][:15]}..."
        ))
        fig_battle.update_layout(template=template, **layout_bg, title="<b>⚔️ HEAD-TO-HEAD BATTLE: TOP 2 APPS</b>", polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
        charts['battle_mode'] = fig_battle.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        charts['battle_mode'] = "<div>Not enough data for Battle Mode</div>"

    fig8 = px.histogram(df, x="Success_Score", nbins=50, color="Type", 
                        title="<b>Success Score Distribution</b>", marginal="box")
    fig8.update_layout(template=template, **layout_bg, barmode="overlay")
    charts['hist_score'] = fig8.to_html(full_html=False, include_plotlyjs='cdn')

    fig9 = px.box(df, x="Category", y="Churn_Rate", color="Category", 
                  title="<b>User Churn Risk Analysis</b>")
    fig9.update_layout(template=template, **layout_bg, showlegend=False)
    charts['box_churn'] = fig9.to_html(full_html=False, include_plotlyjs='cdn')

    fig10 = px.density_heatmap(df, x="Rating", y="Sentiment", facet_col="Type",
                               title="<b>Rating vs. Sentiment Density</b>")
    fig10.update_layout(template=template, **layout_bg)
    charts['heatmap_density'] = fig10.to_html(full_html=False, include_plotlyjs='cdn')

    # Funnel
    funnel_data = dict(
        number=[len(df), len(df[df['Installs']>1000]), len(df[df['Rating']>4.0]), len(df[df['Cluster_Label']=='Market Titans'])],
        stage=["Total Apps", "Active (>1k)", "High Quality (>4.0)", "Market Titans (AI)"]
    )
    fig12 = px.funnel(funnel_data, x='number', y='stage', title="<b>Market Filtration Funnel</b>")
    fig12.update_layout(template=template, **layout_bg)
    charts['funnel'] = fig12.to_html(full_html=False, include_plotlyjs='cdn')

    return charts

# --- 4. THE COMMAND CENTER (HTML GENERATOR) ---
def build_command_center():
    df, df_trends, df_keywords = fetch_massive_data()
    charts = create_command_center_charts(df, df_trends, df_keywords)
    
    # KPIs
    total_users = df['Installs'].sum()
    active_users = df['Daily_Active_Users'].sum()
    avg_score = df['Success_Score'].mean()
    sys_health = 100 - (df['Server_Latency_ms'].mean() / 10)
    
    # Top 1000 Ledger
    print_system_msg("COMPILING TOP 1000 APPS LEDGER...", "INFO")
    top_apps_df = df.sort_values(by='Installs', ascending=False).head(1000)
    
    table_rows = ""
    for index, row in top_apps_df.iterrows():
        # Dynamic color for Success Score
        score_color = "#00ff9d" if row['Success_Score'] > 80 else "#ffcc00" if row['Success_Score'] > 50 else "#ff0055"
        
        table_rows += f"""
        <tr>
            <td style="color:white; font-weight:bold;">{index+1}</td>
            <td>{row['App']}</td>
            <td>{row['Category']}</td>
            <td style="color:#00ff9d">{row['Installs']:,}</td>
            <td>{row['Rating']}</td>
            <td><span style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">{row['Cluster_Label']}</span></td>
            <td style="color:{score_color}; font-weight:bold">{row['Success_Score']:.1f}%</td>
        </tr>
        """
    
    csv_data = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv_data.encode()).decode()
    
    print_system_msg("ASSEMBLING UI INTERFACE...", "INFO")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GOOGLE PLAY ANALYTICS | AI COMMAND CENTER</title>
        <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            :root {{
                --bg-dark: #0a0b10;
                --panel-bg: rgba(20, 25, 40, 0.7);
                --neon-blue: #00f3ff;
                --neon-red: #ff0055;
                --neon-green: #00ff9d;
                --neon-purple: #bd00ff;
                --text-main: #e0e6ed;
            }}
            
            body {{
                font-family: 'Rajdhani', sans-serif;
                background-color: var(--bg-dark);
                background-image: 
                    linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
                background-size: 30px 30px;
                color: var(--text-main);
                margin: 0;
                padding: 20px;
                overflow-x: hidden;
            }}

            /* --- HEADER --- */
            .hud-header {{
                display: flex; justify-content: space-between; align-items: center;
                border-bottom: 2px solid var(--neon-blue); padding-bottom: 15px; margin-bottom: 30px;
            }}
            .brand h1 {{ margin: 0; font-size: 3em; letter-spacing: 5px; text-shadow: 0 0 10px var(--neon-blue); }}
            
            .system-status {{ text-align: right; display: flex; gap: 30px; }}
            .status-val {{ font-size: 1.5em; font-weight: bold; color: var(--neon-green); }}
            .status-lbl {{ font-size: 0.8em; color: #8899a6; }}
            
            .btn-download {{
                background: transparent; border: 1px solid var(--neon-blue); color: var(--neon-blue);
                padding: 10px 25px; font-weight: bold; cursor: pointer; transition: 0.3s;
            }}
            .btn-download:hover {{ background: var(--neon-blue); color: black; box-shadow: 0 0 20px var(--neon-blue); }}

            /* --- KPIs --- */
            .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
            .kpi-card {{
                background: var(--panel-bg); border: 1px solid rgba(0, 243, 255, 0.2); padding: 20px; position: relative;
            }}
            .kpi-card::before {{ content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--neon-blue); }}
            .kpi-value {{ font-size: 2.5em; font-weight: 700; margin-bottom: 5px; }}
            .kpi-label {{ color: #8899a6; font-size: 0.9em; letter-spacing: 1px; }}

            /* --- CHARTS --- */
            .section-title {{ font-size: 1.5em; border-left: 5px solid var(--neon-red); padding-left: 15px; margin: 40px 0 20px 0; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
            .chart-panel {{
                background: var(--panel-bg); border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px; padding: 10px; height: 400px; box-shadow: 0 0 15px rgba(0,0,0,0.5);
            }}
            .full-width {{ grid-column: span 2; }}

            /* --- TABLE --- */
            .search-bar {{ width: 100%; padding: 10px; background: rgba(0,0,0,0.5); border: 1px solid var(--neon-blue); color: white; margin-bottom: 10px; font-family: 'Rajdhani'; font-size: 1.2em; }}
            .table-container {{
                background: var(--panel-bg); border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px; padding: 20px; height: 500px; overflow-y: scroll;
            }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; border-bottom: 2px solid var(--neon-blue); padding: 10px; color: var(--neon-blue); position: sticky; top: 0; background: #0a0b10; }}
            td {{ border-bottom: 1px solid rgba(255,255,255,0.1); padding: 8px; font-size: 0.9em; color: #ccc; }}
            tr:hover {{ background: rgba(0, 243, 255, 0.1); }}

            /* --- ANIMATIONS & LOG --- */
            .log-panel {{
                background: black; border: 1px solid #333; height: 150px; overflow-y: hidden;
                font-family: 'Courier New', monospace; font-size: 0.8em; color: var(--neon-green); padding: 10px; margin-top: 30px; opacity: 0.8;
            }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
            .live-dot {{ height: 10px; width: 10px; background-color: var(--neon-red); border-radius: 50%; display: inline-block; animation: pulse 1s infinite; margin-right: 5px; }}
        </style>
    </head>
    <body onload="initDashboard()">

        <div class="hud-header">
            <div class="brand">
                <h1>GOOGLE PLAY <span style="color:white">ANALYTICS AI</span></h1>
                <span>INTELLIGENT MARKET SYSTEM v6.0</span>
            </div>
            <div class="system-status">
                <div class="status-item">
                    <span class="status-val" id="clock">00:00:00</span>
                    <span class="status-lbl">UTC TIME</span>
                </div>
                <div class="status-item">
                    <span class="status-val" style="color:var(--neon-green)">ACTIVE</span>
                    <span class="status-lbl">AI ENGINE</span>
                </div>
                <div class="status-item">
                    <span class="status-val">{len(df)}</span>
                    <span class="status-lbl">NODES</span>
                </div>
                <button class="btn-download" onclick="downloadCSV()">DOWNLOAD CSV</button>
            </div>
        </div>

        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-value">{total_users/1e9:.2f}B</div>
                <div class="kpi-label">TOTAL MARKET INSTALLS</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color:var(--neon-green)">${active_users * 0.15:,.0f}</div>
                <div class="kpi-label">EST. REAL-TIME REVENUE</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color:var(--neon-purple)">{avg_score:.1f}%</div>
                <div class="kpi-label">AVG SUCCESS PROBABILITY</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color:var(--neon-blue)">{sys_health:.1f}%</div>
                <div class="kpi-label">SYSTEM INTEGRITY</div>
            </div>
        </div>

        <div class="section-title">SECTOR A: GLOBAL TOPOGRAPHY</div>
        <div class="chart-grid">
            <div class="chart-panel">{charts['market_sunburst']}</div>
            <div class="chart-panel">{charts['revenue_treemap']}</div>
        </div>

        <div class="section-title"><span class="live-dot"></span> SECTOR B: PREDICTIVE INTELLIGENCE (ML)</div>
        <div class="chart-grid">
            <div class="chart-panel">{charts['ml_clusters']}</div>
            <div class="chart-panel">{charts['keywords']}</div>
            <div class="chart-panel full-width">{charts['funnel']}</div>
        </div>

        <div class="section-title">SECTOR C: LIVE DATA STREAMS</div>
        <div class="chart-grid">
            <div class="chart-panel full-width">{charts['trend_installs']}</div>
            <div class="chart-panel full-width">{charts['trend_sentiment']}</div>
        </div>

        <div class="section-title">SECTOR D: COMPETITIVE & DEEP METRICS</div>
        <div class="chart-grid">
            <div class="chart-panel">{charts['battle_mode']}</div>
            <div class="chart-panel">{charts['3d_perf']}</div>
            <div class="chart-panel">{charts['hist_score']}</div>
            <div class="chart-panel">{charts['box_churn']}</div>
            <div class="chart-panel full-width">{charts['heatmap_density']}</div>
        </div>

        <div class="section-title">SECTOR E: TOP 1000 SEARCHABLE LEDGER</div>
        <input type="text" id="searchInput" class="search-bar" onkeyup="filterTable()" placeholder="Search for App Name or Category...">
        <div class="table-container">
            <table id="appTable">
                <thead>
                    <tr>
                        <th>RANK</th>
                        <th>APP NAME</th>
                        <th>CATEGORY</th>
                        <th>INSTALLS</th>
                        <th>RATING</th>
                        <th>AI CLUSTER</th>
                        <th>SUCCESS SCORE</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="log-panel" id="console-log">
            > SYSTEM INITIALIZED.<br>
            > CONNECTED TO GOOGLE PLAY API NODES [US-EAST-1]...<br>
        </div>

        <script>
            // FILTER TABLE FUNCTION
            function filterTable() {{
                var input, filter, table, tr, td, i, txtValue;
                input = document.getElementById("searchInput");
                filter = input.value.toUpperCase();
                table = document.getElementById("appTable");
                tr = table.getElementsByTagName("tr");
                for (i = 0; i < tr.length; i++) {{
                    td = tr[i].getElementsByTagName("td")[1]; // Search App Name
                    td2 = tr[i].getElementsByTagName("td")[2]; // Search Category
                    if (td || td2) {{
                        txtValue = (td.textContent || td.innerText) + (td2.textContent || td2.innerText);
                        if (txtValue.toUpperCase().indexOf(filter) > -1) {{
                            tr[i].style.display = "";
                        }} else {{
                            tr[i].style.display = "none";
                        }}
                    }}       
                }}
            }}

            // LIVE CLOCK
            function updateClock() {{
                const now = new Date();
                document.getElementById('clock').innerText = now.toLocaleTimeString();
            }}
            setInterval(updateClock, 1000);

            // SIMULATED LIVE LOG
            const messages = [
                "Running K-Means Clustering on new batch...",
                "Optimizing NLP Sentiment vectors...",
                "Detected Trend Spike in 'Productivity'...",
                "Calculating Success Probabilities...",
                "User node #882 connected...",
                "Revenue stream sync complete.",
            ];
            
            function updateLog() {{
                const log = document.getElementById('console-log');
                const msg = messages[Math.floor(Math.random() * messages.length)];
                const timestamp = new Date().toLocaleTimeString();
                log.innerHTML += `> [${{timestamp}}] ${{msg}}<br>`;
                log.scrollTop = log.scrollHeight;
            }}
            setInterval(updateLog, 1500);

            // DOWNLOAD CSV
            function downloadCSV() {{
                const b64 = "{b64_csv}";
                const blob = new Blob([atob(b64)], {{ type: 'text/csv' }});
                const link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = "google_play_ai_analytics.csv";
                link.click();
            }}

            function initDashboard() {{
                console.log("NEXUS ONLINE");
            }}
        </script>
    </body>
    </html>
    """
    
    filename = "google_playstore_data_analytics.html"
    with open(filename, "w", encoding='utf-8') as f:
        f.write(html)
    
    print("\n")
    print_system_msg("DASHBOARD COMPILED SUCCESSFULLY.", "OK")
    print_system_msg(f"LAUNCHING INTERFACE: {filename}", "OK")

if __name__ == "__main__":
    build_command_center()