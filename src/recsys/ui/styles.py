"""Ultra-Clean Blue & White Design System for CineFlow AI."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@600;700;800&display=swap');

/* =========================================================================
   GLOBAL BLUE & WHITE THEME VARIABLES
   ========================================================================= */
:root {
    --bg-main: #f8fafc;
    --bg-surface: #ffffff;
    --primary-blue: #2563eb;
    --primary-gradient: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%);
    --accent-cyan: #0284c7;
    --accent-ice: #eff6ff;
    --accent-gold: #d97706;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-light: #e2e8f0;
    --border-blue: #bfdbfe;
    --card-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
    --card-hover-shadow: 0 20px 35px -5px rgba(37, 99, 235, 0.18), 0 10px 15px -5px rgba(37, 99, 235, 0.1);
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

/* Background Ambient Gradient */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, #dbeafe 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 30%, #e0f2fe 0%, transparent 50%),
        radial-gradient(ellipse 50% 30% at 0% 70%, #eff6ff 0%, transparent 60%),
        #f8fafc !important;
    background-attachment: fixed !important;
}

/* Hide Default Streamlit Chrome */
#MainMenu, header, footer, .stDeployButton, [data-testid="stToolbar"] {
    display: none !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
}

/* =========================================================================
   TOP NAVIGATION BAR (CRISP WHITE & BLUE)
   ========================================================================= */
.cine-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.75rem;
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 25px rgba(37, 99, 235, 0.08);
}

.cine-logo-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.cine-logo-icon {
    width: 38px;
    height: 38px;
    background: var(--primary-gradient);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
}

.cine-logo-text {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text-primary);
}

.cine-logo-text span {
    color: var(--primary-blue);
}

.cine-badge-live {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #059669;
}

.cine-pulse-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* =========================================================================
   SPOTLIGHT HERO (DEEP ROYAL SAPPHIRE & WHITE)
   ========================================================================= */
.spotlight-hero {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 40%, #2563eb 100%);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 20px 45px -10px rgba(30, 58, 138, 0.35);
    color: #ffffff;
}

.spotlight-hero::before {
    content: "";
    position: absolute;
    top: 0; right: 0; bottom: 0; left: 0;
    background: radial-gradient(circle at 85% 20%, rgba(255, 255, 255, 0.2) 0%, transparent 60%);
    pointer-events: none;
}

.spotlight-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 14px;
    border-radius: 30px;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

.spotlight-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
    color: #ffffff;
}

.spotlight-desc {
    font-size: 1.15rem;
    color: #e0e7ff;
    max-width: 750px;
    line-height: 1.6;
    margin-bottom: 1.75rem;
}

.spotlight-stats-row {
    display: flex;
    gap: 2.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    padding-top: 1.5rem;
}

.spotlight-stat-item {
    display: flex;
    flex-direction: column;
}

.spotlight-stat-num {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
}

.spotlight-stat-lbl {
    font-size: 0.75rem;
    font-weight: 600;
    color: #bfdbfe;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* =========================================================================
   LUXURY BLUE & WHITE MOVIE CARDS
   ========================================================================= */
.cine-card {
    position: relative;
    background: #ffffff;
    border: 1px solid var(--border-light);
    border-radius: 18px;
    padding: 0.75rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: var(--card-shadow);
    overflow: hidden;
}

.cine-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: var(--border-blue);
    box-shadow: var(--card-hover-shadow);
}

.cine-poster-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 3;
    border-radius: 12px;
    overflow: hidden;
    background: #e2e8f0;
    margin-bottom: 0.85rem;
}

.cine-poster-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 12px;
    transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

.cine-card:hover .cine-poster-img {
    transform: scale(1.06);
}

/* Floating Poster Badges */
.cine-badge-rank {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 8px;
    backdrop-filter: blur(8px);
}

.cine-badge-match {
    position: absolute;
    top: 10px;
    right: 10px;
    background: var(--primary-gradient);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

.cine-badge-score {
    position: absolute;
    top: 10px;
    right: 10px;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
}

.cine-movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.7rem;
}

.cine-movie-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 0.65rem;
}

.cine-rating-gold {
    color: #d97706;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 3px;
}

.cine-genre-chip {
    display: inline-block;
    background: var(--accent-ice);
    border: 1px solid var(--border-blue);
    color: #1d4ed8;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    margin-right: 4px;
    margin-bottom: 4px;
    text-transform: capitalize;
}

/* =========================================================================
   EXPLAINABILITY CARD
   ========================================================================= */
.cine-explain-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 0.85rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    line-height: 1.45;
    color: #166534;
}

/* =========================================================================
   STREAMLIT FORM WIDGETS OVERRIDES (BLUE & WHITE)
   ========================================================================= */
.stButton > button {
    background: var(--primary-gradient) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.75rem !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.45) !important;
}

/* Inputs & Selectboxes */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: var(--primary-blue) !important;
}

.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #ffffff !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: var(--primary-blue) !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
}

/* Radio Navigation Bar */
div[data-testid="stRadio"] > div {
    background: #ffffff;
    border: 1px solid var(--border-light);
    padding: 6px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
}

div[data-testid="stRadio"] label {
    padding: 6px 14px;
    border-radius: 10px;
    font-weight: 600;
    color: var(--text-secondary);
    transition: all 0.2s ease;
}

div[data-testid="stRadio"] label:hover {
    color: var(--primary-blue);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border-light) !important;
}
</style>
"""
