import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import os
import plotly.graph_objects as go
st.set_page_config(layout="wide", page_title="Company Dashboard")

# --- LOGIN SYSTEM ---
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Example login credentials
        if username == "sales12@gmail.com" and password == "manager26":
            st.session_state["authenticated"] = True
            st.session_state["username"] = username  # Store username
            st.success("Login successful!")
            st.rerun()  # This will reload the app after login
        else:
            st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()  # Stop further execution if not logged in

# --- SIDEBAR NAVIGATION ---
data = pd.read_csv("rrevamp.csv")  # Load your data
@st.cache_data
def load_data():
    df = pd.read_csv("rrevamp.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Year"] = df["Timestamp"].dt.year  # ✅ Ensure 'Year' column exists
    return df

data = load_data()


with st.sidebar:
    # Logo
    col1, col2, col3 = st.columns([1,2,1])
    


    # Navigation menu
 

    # Divider


    # User info
    st.markdown("### 👤 User Info")
    st.write(f"**Sales Manager:** {st.session_state.get('Prince M Galekhutle', 'Galekhutle')}")    
    # You could also add role if you store it:
    # st.write(f"**Role:** {st.session_state.get('role','N/A')}")

    # Logout button
    if st.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()

 

    

# Inject blue hover CSS for option_menu
    st.markdown("""
        <style>
        .nav-pills .nav-link {
            color: #004080;
        }
        .nav-pills .nav-link:hover {
            background-color: #0d6efd !important;  /* Blue hover */
            color: white !important;
        }
        .nav-pills .nav-link.active {
            background-color: #0d6efd !important;  /* Blue for active tab */
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Your menu
    selected = option_menu(
        menu_title="Dashboard",
        options=["Sales Overview", "Perfomance","Marketing", "AI Model", "Notifications"],
        icons=["graph-up", "people", "bullseye", "bullhorn" ],
        menu_icon="columns-gap",
        default_index=0
    )


    

    # Sales Member Filter for the Sales Team
    
        

        
        
    


# Page Renderer



# Top summary metrics
    


    

    # Prepare Country Data
data['Country'] = data['Country'].astype(str).str.strip()
country_counts = data['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Sessions']

# --- Sidebar Filters ---



if selected == "Sales Overview":
    # 1) Global CSS to tighten up spacing
    st.markdown("""
    <style>
        .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        }
        /* reduce gaps between columns */
        .css-1lcbmhc { gap: 0.1rem !important; }
        /* reduce space below each chart */
        .stContainer { margin-bottom: 0.1rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # --- Header & Sidebar Filters ---
    
    

    st.sidebar.markdown("### Filters")
    selected_country = st.sidebar.multiselect("Select Country", data['Country'].unique())
    selected_year = st.sidebar.multiselect("Select Year", sorted(data['Timestamp'].dt.year.dropna().unique()))
    
    filtered_data = data.copy()
    total_revenue    = filtered_data['ProductSales'].sum()
    total_purchases  = filtered_data[filtered_data['ActivityType']=='Purchase'].shape[0]
    total_views      = filtered_data[filtered_data['ActivityType']=='View'].shape[0]
    conv_rate        = (total_purchases / total_views * 100) if total_views else 0
    filtered_data = data.copy()
    if selected_country:
     filtered_data = filtered_data[filtered_data['Country'].isin(selected_country)]
    if selected_year:
     filtered_data = filtered_data[filtered_data['Timestamp'].dt.year.isin(selected_year)]
     total_revenue    = filtered_data['ProductSales'].sum()
     total_purchases  = filtered_data[filtered_data['ActivityType']=='Purchase'].shape[0]
     total_views      = filtered_data[filtered_data['ActivityType']=='View'].shape[0]
     total_ad_clicks  = filtered_data['AdClicks'].sum() if 'AdClicks' in filtered_data.columns else 0
     avg_session      = filtered_data['SessionDuration(min)'].mean() if 'SessionDuration(min)' in filtered_data.columns else 0
     conv_rate        = (total_purchases / total_views * 100) if total_views else 0


    st.markdown(f"""
    <style>
    .kpi-container {{
        border-bottom: 2px solid #fff;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }}
    .kpi-row {{
        display: flex;
        justify-content: space-around;
        font-size: 1.0rem;
        font-weight: 300;
        color: #333;
    }}
    .kpi-item {{
        text-align: center;
        font-size: 0.85rem;
        font-weight: bold;
        color: white;
        text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.8);
        margin: 0.5rem;
    }}
    </style>

    <div class="kpi-container">
        <div class="kpi-row">
            <div class="kpi-item">👀<br><strong>{total_views:,}</strong><br>Views</div>
            <div class="kpi-item">🛒<br><strong>{total_purchases:,}</strong><br>Purchases</div>
            <div class="kpi-item">🖱️<br><strong>{total_views:,}</strong><br>Ad Clicks</div>
            <div class="kpi-item">⏱️<br><strong>{conv_rate:.1f}</strong><br>Avg. Session</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
 




    


    
    
    # … your existing header, sidebar filters and data prep …

    # --- KPI PANEL ---
   
    # --- Chart 1: Product‐Revenue Choropleth (fig1) ---
    # … your revised fig1 code from earlier …

    # --- Chart 2: Product Views vs Purchases (fig7) ---
    # … your fig7 code …

    # --- Render Top Row ---
    

    # (we’ve removed the old yearly‐sales count chart entirely)

    # … now continue with bottom‐row charts (fig5, fig6) as before …


    # --- Chart 1: Product‐Revenue Choropleth (fig1) ---
    # … your revised fig1 code from earlier …

    # --- Chart 2: Product Views vs Purchases (fig7) ---
    # … your fig7 code …

    # --- Render Top Row ---
    

    # (we’ve removed the old yearly‐sales count chart entirely)

    # … now continue with bottom‐row charts (fig5, fig6) as before …


    # --- Apply Filters ---
    filtered_data = data.copy()
    filtered_data['Year'] = filtered_data['Timestamp'].dt.year

    if selected_country:
        filtered_data = filtered_data[filtered_data['Country'].isin(selected_country)]
    if selected_year:
        filtered_data = filtered_data[filtered_data['Year'].isin(selected_year)]

    # --- Clean & Prep ---
    filtered_data['Country'] = filtered_data['Country'].astype(str).str.strip()
    purchases = filtered_data[filtered_data['ActivityType'] == 'Purchase']

    # --- Performance Color Setup ---
    # Define average deal size targets by country
    avg_deal_targets_by_country = {
        'United States': 250,
        'India': 250,
        'United Kingdom': 250,
        'Germany': 250,
        'Canada': 1000,
        'Australia': 1900000,
    }
    default_avg_deal_target = 50  # fallback target

    # Calculate average deal size by country (purchases only)
    

    avg_deal_size = (
        purchases
        .groupby('Country')['ProductSales']
        .mean()
        .reset_index(name='AvgDealSize')
    )

    # Assign target for each country
    avg_deal_size['Target'] = avg_deal_size['Country'].apply(
        lambda c: avg_deal_targets_by_country.get(c, default_avg_deal_target)
    )

    # Performance category function (same as before)
    def perf_category(val, tgt):
        if val >= tgt:
            return 'Met Target'
        elif val >= 0.8 * tgt:
            return 'Near Target'
        else:
            return 'Below Target'

    # Apply performance category
    avg_deal_size['PerfCat'] = avg_deal_size.apply(
        lambda row: perf_category(row['AvgDealSize'], row['Target']), axis=1
    )

    # Color mapping for performance
    perf_colour_map = {
        'Met Target': "#0D0876",
        'Near Target': "#217098",
        'Below Target': "#76BDDB",
    }

    # Choropleth Map: Performance by Avg Deal Size
    fig1 = px.choropleth(
        avg_deal_size,
        locations='Country',
        locationmode='country names',
        color='PerfCat',
        hover_name='Country',
        hover_data={'AvgDealSize': ':.2f', 'Target': True},
        color_discrete_map=perf_colour_map,
        title='🌍 Average Deal Size by Country',
        template='plotly_white',
        projection='natural earth',
        height=300
    )

    fig1.update_layout(
        legend=dict(orientation='v', x=1.02, y=1)
    )

    # --- Chart 2: Revenue by Product ID ---
    # --- Define target per product or use a constant ---
    product_targets = {
        'P001': 7000000, 'P002':7000000, 'P003': 7000000,
        'P004': 7000000, 'P005': 7000000, 'P006': 7000000
    }
    default_target = 7000000  # fallback if not in dict

    # --- Compute revenue per ProductID ---
    df_revenue = (
        filtered_data
        .groupby('ProductID')['ProductSales']
        .sum()
        .reset_index(name='TotalRevenue')
    )

    # --- Assign target and performance category ---
    product_revenue = filtered_data.groupby('ProductID')['ProductSales'].sum().reset_index()
    fig2 = px.pie(
            product_revenue,
            names='ProductID',
            values='ProductSales',
            title='💼 Revenue Distribution by Product (Ring Chart)',
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
    fig2.update_layout(height=300)
    


    # --- Render Top Row ---
    with st.container():
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig1, use_container_width=True)
        with c2: st.plotly_chart(fig2, use_container_width=True)

    # --- Chart 3: Total Sales by Country (Yearly) ---
    country_year_target = 5000
    df3 = (
        purchases
        .groupby(['Year', 'Country'])
        .size()
        .reset_index(name='TotalSales')
    )
    df3['PerfCat'] = df3['TotalSales'].apply(lambda v: perf_category(v, country_year_target))

    fig3 = go.Figure()
    for country, dfi in df3.groupby('Country'):
        fig3.add_trace(go.Scatter(
            x=dfi['Year'], y=dfi['TotalSales'],
            mode='lines+markers',
            name=country,
            line=dict(dash='dash', color=perf_colour_map[dfi['PerfCat'].iloc[-1]]),
            marker=dict(size=8),
        ))

    for name, color in perf_colour_map.items():
        fig3.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                  marker_color=color, name=name))

    fig3.update_layout(
        title="📈 Total Sales by Country (Yearly)",
        template="plotly_white",
        height=300,
        legend=dict(orientation='v', x=1.02, y=1)
    )

    # --- Chart 4: Revenue per Country ---
    revenue_target = 5000000
    df4 = (
        filtered_data
        .groupby('Country')['ProductSales']
        .sum()
        .reset_index(name='TotalRevenue')
    )
    df4['PerfCat'] = df4['TotalRevenue'].apply(lambda v: perf_category(v, revenue_target))

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=df4['Country'],
        y=df4['TotalRevenue'],
        marker_color=df4['PerfCat'].map(perf_colour_map),
        name='Revenue'
    ))

    avg_rev = df4['TotalRevenue'].mean()
    fig4.add_shape(
        type='line',
        x0=-0.5, x1=len(df4['Country']) - 0.5,
        y0=avg_rev, y1=avg_rev,
        line=dict(color='black', dash='dash'),
        xref='x', yref='y'
    )
    fig4.add_annotation(
        x=len(df4['Country']) - 1,
        y=avg_rev,
        text=f"Avg: {avg_rev:,.0f}",
        showarrow=False,
        yshift=10,
        font=dict(color='black')
    )

    for name, color in perf_colour_map.items():
        fig4.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10),
            name=name
        ))

    fig4.update_layout(
        title='💰 Revenue per Country',
        xaxis_title='Country',
        yaxis_title='Total Revenue',
        template='plotly_white',
        height=320,
        legend=dict(orientation='v', x=1.02, y=1)
    )

    # --- Chart 5: Yearly Sales Count (Performance) ---
    overall_target = 240000  # realistic
    df5 = (
        purchases
        .groupby('Year')
        .size()
        .reset_index(name='TotalSales')
    )
    df5['PerfCat'] = df5['TotalSales'].apply(lambda v: perf_category(v, overall_target))

    fig5 = px.bar(
        df5,
        x='Year',
        y='TotalSales',
        text='TotalSales',
        color='PerfCat',
        color_discrete_map=perf_colour_map,
        title="📊 Yearly Sales Count (Performance)",
        template="plotly_white",
        height=300
    )
    fig5.add_shape(
        type='line',
        x0=df5['Year'].min() - 0.4,
        x1=df5['Year'].max() + 0.4,
        y0=overall_target,
        y1=overall_target,
        line=dict(color='black', dash='dash'),
    )
    fig5.update_traces(textposition='outside')
    fig5.update_layout(legend=dict(orientation='v', x=1.02, y=1))

    
# --- Sales Performance Tracking Chart (New Chart) ---
    monthly_target = 70000  # Adjust as needed
    import pandas as pd

# Load your data
    df = pd.read_csv("rrevamp.csv")  # Replace with the correct path if needed

    # Any preprocessing steps here (optional)
    df.dropna(subset=["ActivityType", "SalesMember"], inplace=True)

    # Group purchases by sales member
    performance_df = df[df["ActivityType"] == "Purchase"].groupby("SalesMember")["ActivityType"].count().reset_index()
    performance_df.rename(columns={"ActivityType": "ActualPurchases"}, inplace=True)
    performance_df["Target"] = monthly_target

    # Bar chart showing actual vs target per sales member
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=performance_df["SalesMember"],
        y=performance_df["ActualPurchases"],
        name="Actual",
        marker_color='lightskyblue'
    ))
    fig6.add_trace(go.Bar(
        x=performance_df["SalesMember"],
        y=performance_df["Target"],
        name="Target",
        marker_color='lightgray'
    ))
    fig6.update_layout(
        barmode="group",
        title="Sales Member Performance vs Target",
        xaxis_title="Sales Member",
        yaxis_title="Purchases",
        height=300,
        margin=dict(t=100, b=90)
    )

    # --- CHART ROW 2: 3 Charts Side-by-Side ---
    c3, c4, c5 = st.columns([1, 1, 1])
    with c3:
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        st.plotly_chart(fig4, use_container_width=True)
    with c5:
        st.plotly_chart(fig6, use_container_width=True)

   


    

    # --- Render Bottom Row ---
   
        
    






# … your other code and imports …


# … your other code and imports …


if selected == "Perfomance":
    st.markdown("""
    <h1 style="font-family: 'Arial', sans-serif; font-size: 1.6rem; color: white; margin-bottom: 0.25rem;">
        📊 Sales Performance
    </h1>
""", unsafe_allow_html=True)


    st.sidebar.markdown("### 📊 Sales Team Filters")
    years = sorted(data['Year'].dropna().unique())
    selected_year = st.sidebar.selectbox("Year", ["All"] + years)
    countries = sorted(data['Country'].dropna().unique())
    selected_country = st.sidebar.selectbox("Country", ["All"] + countries)
    sales_members = sorted(data['SalesMember'].dropna().unique())
    sales_member_filter = st.sidebar.selectbox("Sales Member", ["All"] + sales_members)

    # --- Targets ---
    yearly_targets  = {2023: 6000, 2024: 4000, 2025: 7000}
    revenue_targets = {2023: 500000, 2024: 750000, 2025: 100000}
    monthly_target  = 700

    # --- Filter Data ---
    fd = data.copy()
    if selected_year != "All":
        fd = fd[fd.Year == selected_year]
    if selected_country != "All":
        fd = fd[fd.Country == selected_country]
    if sales_member_filter != "All":
        fd = fd[fd.SalesMember == sales_member_filter]

    fd['Timestamp']    = pd.to_datetime(fd['Timestamp'], errors='coerce')
    fd['ProductSales'] = pd.to_numeric(fd['ProductSales'], errors='coerce').fillna(0)
    purchases = fd[fd.ActivityType == 'Purchase'].dropna(subset=['Timestamp'])

    # --- KPI Calculations ---
    total_purchases  = len(purchases)
    total_revenue    = purchases.ProductSales.sum()
    avg_per_order    = total_revenue / total_purchases if total_purchases else 0
    unique_customers = purchases.SalesMember.nunique()
    best_member = (
        purchases.groupby('SalesMember').ProductSales.sum().idxmax()
        if not purchases.empty else "N/A"
    )

    # --- Custom KPI Card ---
    def custom_kpi(label, value):
        st.markdown(f"""
            <div style="
                background: none;
                padding: 4px 8px;
                border-radius: 4px;
                text-align: center;
                font-size: 14px;
                margin-bottom:10px;
            ">
                <div style="font-weight:600; font-size:13px;">{label}</div>
                <div style="font-size:16px; margin-top:2px;"><strong>{value}</strong></div>
            </div>
        """, unsafe_allow_html=True)

    # --- Display KPIs in 4 columns ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: custom_kpi("🛒 Total Sales", f"{total_purchases:,}")
    with c2: custom_kpi("💰 Total Revenue", f"${total_revenue:,.0f}")
    with c3: custom_kpi("📈 Avg per Order", f"${avg_per_order:,.2f}")
    with c4: custom_kpi("👥 Active Members", f"{unique_customers:,}")

    # --- Top Performer Alert (compact) ---
    progress = total_revenue / sum(revenue_targets.values())
    if progress > 0.9:
        color = "#026f1ac6"
        emoji = "🚀"
        msg = f"{emoji} {progress:.0%} of all‑years revenue target reached!"
    elif progress > 0.7:
        color = "#834e0a"
        emoji = "ℹ️"
        msg = f"{emoji} {progress:.0%} of all‑years revenue goal. Keep pushing!"
    else:
        color = "#7b0606"
        emoji = "⚠️"
        msg = f"{emoji} Only {progress:.0%} of revenue goal—ramp up efforts."

    st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 6px 10px;
            border-radius: 10px;
            font-size: 13px;
            margin-top: 10px;
            margin-bottom: 10px;
        ">{msg}</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="
            background-color: #081aa2;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 10px;
        ">🏅 <strong>Top Sales Member:</strong> {best_member}</div>
    """, unsafe_allow_html=True)

    # --- PERFORMANCE COLOR HELPER ---
    def perf_color(ratio):
        if ratio >= 1.0: return "#66E570"
        if ratio >= 0.7: return "#B6BA33"
        return "#DF2222"

    # --- 1) Sales Gauge ---
    if selected_year == "All":
        total_sales  = total_purchases
        target_sales = sum(yearly_targets.values())
        title_sales  = "All‑Years Sales"
    else:
        total_sales  = purchases[purchases.Timestamp.dt.year == selected_year].shape[0]
        target_sales = yearly_targets[selected_year]
        title_sales  = f"Year {selected_year} Sales"

    fig_sales = go.Figure()
    for nm, clr in [("Below 70%", "#AED6F1"), ("70–99%", "#5DADE2"), ("≥100%", "#007ACC")]:
        fig_sales.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                       marker=dict(size=6, color=clr), name=nm))
    fig_sales.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=total_sales, delta={'reference': target_sales},
        title=None,  # Title removed from inside plot
        gauge={
            'axis': {'range': [0, target_sales * 1.5], 'tickfont': {'size': 10}},
            'bar': {'color': "#50F072"},
            'steps': [
                {'range': [0, target_sales * 0.7], 'color': '#AED6F1'},
                {'range': [target_sales * 0.7, target_sales], 'color': '#5DADE2'},
                {'range': [target_sales, target_sales * 1.5], 'color': '#007ACC'}
            ],
            'threshold': {'value': target_sales,
                          'line': {'color': "#79E479", 'width': 1},
                          'thickness': 0.6}
        }
    ))

    # --- 2) Revenue Gauge ---
    if selected_year == "All":
        total_rev  = total_revenue
        target_rev = sum(revenue_targets.values())
        title_rev  = "All‑Years Revenue"
    else:
        mask = purchases.Timestamp.dt.year == selected_year
        total_rev  = purchases.loc[mask, 'ProductSales'].sum()
        target_rev = revenue_targets[selected_year]
        title_rev  = f"Year {selected_year} Revenue"

    fig_rev = go.Figure()
    for nm, clr in [("Below 70%", "#AED6F1"), ("70–99%", "#5DADE2"), ("≥100%", "#007ACC")]:
        fig_rev.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(size=6, color=clr), name=nm))
    fig_rev.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=total_rev,
        number={'prefix': "$", 'valueformat': ',.0f'},
        delta={'reference': target_rev, 'valueformat': ',.0f'},
        title=None,  # Removed internal title
        gauge={
            'axis': {'range': [0, target_rev * 1.5], 'tickprefix': "$", 'tickfont': {'size': 10}},
            'bar': {'color': "#50F072"},
            'steps': [
                {'range': [0, target_rev * 0.7], 'color': '#AED6F1'},
                {'range': [target_rev * 0.7, target_rev], 'color': '#5DADE2'},
                {'range': [target_rev, target_rev * 1.5], 'color': '#007ACC'}
            ],
            'threshold': {'value': target_rev,
                          'line': {'color': "#79E479", 'width': 1},
                          'thickness': 0.6}
        }
    ))

    # --- Display the two gauges with external titles ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📊 {title_sales}", anchor=None)
        fig_sales.update_layout(
            height=150,
            margin={'t': 40, 'b': 10},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with col2:
        st.subheader(f"💵 {title_rev}", anchor=None)
        fig_rev.update_layout(
            height=150,
            margin={'t': 40, 'b': 10},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    # --- 3) Monthly Sales Trend ---
    monthly = purchases.copy()
    monthly['Month'] = monthly.Timestamp.dt.to_period('M')
    mgrp = (
        monthly.groupby('Month')
        .size().reset_index(name='TotalSales')
        .sort_values('Month')
    )
    mgrp['Ratio'] = mgrp['TotalSales'] / monthly_target
    mgrp['Color'] = mgrp['Ratio'].apply(perf_color)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=mgrp['Month'].astype(str),
        y=mgrp['TotalSales'],
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=6, color=mgrp['Color'], line=dict(width=1, color='DarkSlateGrey')),
        name='Monthly Sales'
    ))
    for nm, clr in [('Met Target (≥100%)', 'green'),
                    ('Near Target (80–99%)', 'goldenrod'),
                    ('Below Target (<80%)', 'crimson')]:
        fig3.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                  marker=dict(size=6, color=clr), name=nm))

    fig3.update_layout(
        title=dict(text='📅 Monthly Sales Trend', font=dict(size=25)),
        xaxis=dict(title='Month', tickfont=dict(size=20)),
        yaxis=dict(title='Total Sales', tickfont=dict(size=20)),
        legend=dict(orientation='v', y=1, x=1.02),
        template='plotly_white',
        height=240,
        margin={'t': 50, 'b': 50},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig3, use_container_width=True)

if selected == "Marketing":
    st.title("📣 Marketing Dashboard")

    # -- Filters --
    st.sidebar.markdown("### Marketing Filters")
    years = sorted(data['Year'].dropna().unique())
    sel_year = st.sidebar.multiselect("Year", years)
    jobtypes = sorted(data['JobType'].dropna().unique())
    sel_job = st.sidebar.multiselect("JobType", jobtypes)
    products = sorted(data['ProductID'].dropna().unique())
    sel_prod = st.sidebar.multiselect("Product", products)

    # -- Apply Filters --
    mdf = data.copy()
    if sel_year: mdf = mdf[mdf['Year'].isin(sel_year)]
    if sel_job: mdf = mdf[mdf['JobType'].isin(sel_job)]
    if sel_prod: mdf = mdf[mdf['ProductID'].isin(sel_prod)]

    # -- KPIs --
    total_sessions = mdf.shape[0]
    total_views = mdf[mdf['ActivityType']=='View'].shape[0]
    total_purchases = mdf[mdf['ActivityType']=='Purchase'].shape[0]
    avg_session = mdf['SessionDuration(min)'].mean() if 'SessionDuration(min)' in mdf.columns else 0
    avg_satisfaction = mdf['SatisfactionRating'].mean() if 'SatisfactionRating' in mdf.columns else 0

    st.markdown("""
    <style>
      .kpi-container { display:flex; gap:10px; margin-bottom:10px; }
      .kpi-item { background:#2e2e2e; padding:10px; border-radius:6px; color:white; font-size:0.9rem; text-align:center; flex:1; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kpi-container'>
      <div class='kpi-item'>🎯 Total Sessions<br><strong>{total_sessions:,}</strong></div>
      <div class='kpi-item'>👀 Views<br><strong>{total_views:,}</strong></div>
      <div class='kpi-item'>🛒 Purchases<br><strong>{total_purchases:,}</strong></div>
      <div class='kpi-item'>⏱️ Avg Session<br><strong>{avg_session:.1f} min</strong></div>
      <div class='kpi-item'>⭐ Satisfaction<br><strong>{avg_satisfaction:.1f}/5</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # -- Top Row Charts --
    c1,c2 = st.columns(2)
    # Views per Country
    with c1:
        vm = mdf[mdf['ActivityType']=='View'].groupby('Country').size().reset_index(name='Views')
        figA = px.bar(vm.sort_values('Views',ascending=True), x='Views', y='Country', orientation='h',
                      title='👁️ Views by Country', height=250)
        st.plotly_chart(figA, use_container_width=True)
    # Ring of JobType by Product
    with c2:
        jp = mdf.groupby(['JobType','ProductID']).size().reset_index(name='Count')
        figB = px.pie(jp, names='JobType', values='Count', hole=0.5,
                      title='📦 JobType Distribution by Product', height=250)
        st.plotly_chart(figB, use_container_width=True)

    # -- Bottom Row Charts --
    c3,c4,c5 = st.columns(3)
    # Satisfaction Rate by Country
    with c3:
        sat = mdf.groupby('Country')['SatisfactionRating'].mean().reset_index()
        figC = px.bar(sat.sort_values('SatisfactionRating',ascending=True), x='SatisfactionRating', y='Country', orientation='h',
                      title='😊 Avg Satisfaction by Country', height=250)
        st.plotly_chart(figC, use_container_width=True)
    # ActivityType by Referrer
    with c4:
        ar = mdf.groupby(['Referrer','ActivityType']).size().reset_index(name='Count')
        figD = px.bar(ar, x='Count', y='Referrer', color='ActivityType', orientation='h',
                      title='🔗 Activity by Referrer', height=250)
        st.plotly_chart(figD, use_container_width=True)
    # Views by JobType
    with c5:
        jv = mdf[mdf['ActivityType']=='View'].groupby('JobType').size().reset_index(name='Views')
        figE = px.bar(jv.sort_values('Views',ascending=True), x='Views', y='JobType', orientation='h',
                      title='💼 Views by JobType', height=250)
        st.plotly_chart(figE, use_container_width=True)

    
    
if selected == "AI Model":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

    # Load dataset
    df = pd.read_csv("rrevamp.csv")

    st.markdown("""
    <style>
    .title {
        font-size: 30px;
        font-weight: 600;
        color: #ffffff;
        text-align: center;  /* Centers the title */
        margin-top: 20px;  /* Adds space from the very top */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🧠 Activity Type Prediction (Purchase vs View)</div>', unsafe_allow_html=True)


    # --- Data Preprocessing ---
    df['ActivityType'] = df['ActivityType'].map({'Purchase': 1, 'View': 0})
    df['ResponseCode'] = pd.to_numeric(df['ResponseCode'], errors='coerce')
    df['ProductPrice'] = pd.to_numeric(df['ProductPrice'], errors='coerce').fillna(0)
    df['SessionDuration(min)'] = pd.to_numeric(df['SessionDuration(min)'], errors='coerce')
    df['SatisfactionRating'] = pd.to_numeric(df['SatisfactionRating'], errors='coerce')

    features = ['Country', 'JobType', 'Method', 'ResponseCode', 'ProductPrice', 'SessionDuration(min)', 'SatisfactionRating']
    df = df[features + ['ActivityType']].dropna()

    # Encode categoricals
    label_encoders = {}
    for col in ['Country', 'JobType', 'Method']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[features]
    y = df['ActivityType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Selection ---
    model_choice = st.sidebar.selectbox("Choose a Model", ["Random Forest", "XGBoost"])
    if model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # --- Display metrics ---
    st.subheader(f"🔍 Model: {model_choice}")
    st.markdown(f"**Accuracy:** {acc:.2f}")

    with st.expander("See Full Classification Report"):
        report = classification_report(y_test, preds, output_dict=False)
        st.text(report)

    # --- User Input for Prediction ---
    st.subheader("📊 Predict Activity Type")

    input_data = {}

    # --- First Row: Categorical Selectboxes ---
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['Country'] = st.selectbox("Country", label_encoders['Country'].classes_)
    with col2:
        input_data['JobType'] = st.selectbox("JobType", label_encoders['JobType'].classes_)
    with col3:
        input_data['Method'] = st.selectbox("Method", label_encoders['Method'].classes_)

    # --- Second Row: Numeric Inputs ---
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        input_data['ResponseCode'] = st.number_input("ResponseCode", value=200)
    with col5:
        input_data['ProductPrice'] = st.number_input("ProductPrice", value=0.0)
    with col6:
        input_data['SessionDuration(min)'] = st.number_input("SessionDuration(min)", value=0.0)
    with col7:
        input_data['SatisfactionRating'] = st.number_input("SatisfactionRating", value=0.0)

    # --- Encode input and Predict ---
    input_df = pd.DataFrame([input_data])
    for col in ['Country', 'JobType', 'Method']:
        input_df[col] = label_encoders[col].transform(input_df[col])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Prediction: **Purchase**")
        else:
            st.info("🧐 Prediction: **View**")


if selected == "Notifications":
    st.title("🔔 Sales Notifications Center")

    # --- Milestone Sales Alerts ---
    st.subheader("🏆 Sales Milestone Alerts")
    milestone_target = st.number_input("Set Sales Milestone", min_value=1, value=100, step=10)
    purchase_data = data[data['ActivityType'] == 'Purchase']

    if not purchase_data.empty:
        member_sales = purchase_data.groupby('SalesMember').size().reset_index(name='TotalSales')
        member_sales = member_sales.sort_values(by='TotalSales', ascending=False)

        for _, row in member_sales.iterrows():
            name = row['SalesMember']
            sales = row['TotalSales']
            if sales >= milestone_target:
                st.success(f"🏅 {name} has hit the milestone! ({sales} sales)")
            elif sales >= milestone_target * 0.9:
                st.warning(f"⚠️ {name} is close to the milestone! ({sales} sales)")
            else:
                st.info(f"📉 {name} needs improvement. ({sales} sales)")
    else:
        st.info("No purchase data available to evaluate milestones.")
    st.subheader("🏅 Top Product of the Month")
    purchase_data['Month'] = pd.to_datetime(purchase_data['Timestamp']).dt.to_period('M')
    latest_month = purchase_data['Month'].max()

    top_product = purchase_data[purchase_data['Month'] == latest_month]['ProductID'].value_counts().idxmax()
    top_count = purchase_data[purchase_data['ProductID'] == top_product].shape[0]

    st.success(f"📦 Product `{top_product}` is the top seller this month! ({top_count} sales)")

    st.subheader("📉 Low Activity Alerts")
 
    purchase_data = data[data['ActivityType'] == 'Purchase'].copy()


    if purchase_data.empty:
     st.info("No purchase data available.")
    else:
     purchase_data['Timestamp'] = pd.to_datetime(purchase_data['Timestamp'], errors='coerce')
     purchase_data = purchase_data.dropna(subset=['Timestamp'])

    if purchase_data['Timestamp'].isna().all():
        st.info("No valid timestamps available in purchase data.")
    else:
        recent_days = purchase_data['Timestamp'].max().normalize()
        inactive_cutoff = recent_days - pd.Timedelta(days=7)

        recent_sales = purchase_data[purchase_data['Timestamp'] > inactive_cutoff]
        active_members = recent_sales['SalesMember'].dropna().unique()
        all_members = data['SalesMember'].dropna().unique()

        inactive_members = set(all_members) - set(active_members)

        if inactive_members:
            for member in inactive_members:
                st.error(f"🚫 {member} has no sales in the last 7 days.")
        else:
            st.success("✅ All sales members have activity in the last 7 days.")
            st.subheader("🔥 Monthly Consistency Streaks")

# Ensure proper datetime formatting
            streak_data = purchase_data.copy()
            streak_data['Month'] = pd.to_datetime(streak_data['Timestamp'], errors='coerce').dt.to_period('M')

# Count number of active months per SalesMember
            monthly_streaks = streak_data.groupby(['SalesMember', 'Month']).size().reset_index(name='Sales')
            monthly_counts = monthly_streaks.groupby('SalesMember').size().reset_index(name='ActiveMonths')

# Filter those with good streaks (e.g., at least 3 active months)
            top_monthly_streaks = monthly_counts[monthly_counts['ActiveMonths'] >= 3]

# Display
            if not top_monthly_streaks.empty:
             for _, row in top_monthly_streaks.iterrows():
              st.success(f"🔥 {row['SalesMember']} has made sales in {row['ActiveMonths']} different months!")
            else:
                 st.info("No consistent monthly sales streaks yet.")
                 

   



    



# Convert timestamp column



 



    




    

