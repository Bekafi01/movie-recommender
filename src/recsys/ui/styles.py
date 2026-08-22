"""Ultra-Premium Dark Cinema & Glassmorphism Design System for CineFlow AI."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@600;700;800&display=swap');

/* =========================================================================
   GLOBAL RESET & TYPOGRAPHY
   ========================================================================= */
:root {
    --bg-dark: #07090e;
    --bg-surface: #0e131f;
    --bg-card: rgba(18, 24, 38, 0.7);
    --primary: #8b5cf6;
    --primary-glow: rgba(139, 92, 246, 0.35);
    --accent-pink: #ec4899;
    --accent-cyan: #06b6d4;
    --accent-gold: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-glass: rgba(255, 255, 255, 0.08);
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

/* Background Ambient Lighting */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120, 119, 198, 0.15), transparent),
        radial-gradient(ellipse 60% 40% at 100% 40%, rgba(236, 72, 153, 0.08), transparent),
        radial-gradient(ellipse 50% 30% at 0% 70%, rgba(6, 182, 212, 0.08), transparent),
        #07090e !important;
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
   TOP NAVIGATION BAR
   ========================================================================= */
.cine-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.75rem;
    background: rgba(14, 19, 31, 0.75);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
}

.cine-logo-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.cine-logo-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 4px 15px var(--primary-glow);
}

.cine-logo-text {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.cine-badge-live {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #34d399;
}

.cine-pulse-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* =========================================================================
   CINEMATIC SPOTLIGHT HERO
   ========================================================================= */
.spotlight-hero {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(16px);
}

.spotlight-hero::before {
    content: "";
    position: absolute;
    top: 0; right: 0; bottom: 0; left: 0;
    background: radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.15) 0%, transparent 60%);
    pointer-events: none;
}

.spotlight-tag {
    display: inline-block;
    background: linear-gradient(135deg, #8b5cf6, #ec4899);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 14px;
    border-radius: 30px;
    margin-bottom: 1rem;
}

.spotlight-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.spotlight-desc {
    font-size: 1.15rem;
    color: #cbd5e1;
    max-width: 750px;
    line-height: 1.6;
    margin-bottom: 1.75rem;
}

.spotlight-stats-row {
    display: flex;
    gap: 2.5rem;
    border-top: 1px solid var(--border-glass);
    padding-top: 1.5rem;
}

.spotlight-stat-item {
    display: flex;
    flex-direction: column;
}

.spotlight-stat-num {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.spotlight-stat-lbl {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* =========================================================================
   LUXURY MOVIE CARDS
   ========================================================================= */
.cine-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 18px;
    padding: 0.75rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(12px);
    overflow: hidden;
}

.cine-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 0 0 25px rgba(139, 92, 246, 0.25);
}

.cine-poster-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 3;
    border-radius: 12px;
    overflow: hidden;
    background: #131b2e;
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
    background: rgba(7, 9, 14, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #f8fafc;
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
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

.cine-badge-score {
    position: absolute;
    top: 10px;
    right: 10px;
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 8px;
    box-shadow: 0 4px 12px var(--primary-glow);
}

.cine-movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
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
    color: #94a3b8;
    margin-bottom: 0.65rem;
}

.cine-rating-gold {
    color: #f59e0b;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 3px;
}

.cine-genre-chip {
    display: inline-block;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #cbd5e1;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 6px;
    margin-right: 4px;
    margin-bottom: 4px;
    text-transform: capitalize;
}

/* =========================================================================
   EXPLAINABILITY CARD
   ========================================================================= */
.cine-explain-box {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(236, 72, 153, 0.06) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 0.85rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    line-height: 1.45;
    color: #e2e8f0;
}

.cine-explain-tag {
    display: inline-block;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #c084fc;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 4px;
    margin-bottom: 3px;
}

/* =========================================================================
   STREAMLIT FORM WIDGET OVERRIDES (SLIDERS, BUTTONS, SELECTBOXES)
   ========================================================================= */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 50%, #ec4899 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.75rem !important;
    box-shadow: 0 4px 20px var(--primary-glow) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(217, 70, 239, 0.45) !important;
}

/* Custom Selectbox & Inputs */
div[data-baseweb="select"] > div {
    background-color: rgba(18, 24, 38, 0.85) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
}

.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: rgba(18, 24, 38, 0.85) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    margin-top: 0.5rem !important;
}

/* Radio / Tabs Navigation Bar */
div[data-testid="stRadio"] > div {
    background: rgba(18, 24, 38, 0.6);
    border: 1px solid var(--border-glass);
    padding: 6px;
    border-radius: 16px;
    backdrop-filter: blur(12px);
}

div[data-testid="stRadio"] label {
    padding: 6px 14px;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid var(--border-glass) !important;
}
</style>
"""
