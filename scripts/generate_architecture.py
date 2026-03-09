"""
TradeSimple Architecture Diagram — clean flowchart
Run: python scripts/generate_architecture.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), '..', 'architecture.png')

C = dict(
    bg      = '#FFFFFF',
    infra   = '#1D4ED8',
    node    = '#4338CA',
    hitl    = '#D97706',
    hitl_lt = '#FFFBEB',
    skip    = '#15803D',
    err     = '#DC2626',
    gray    = '#6B7280',
    lt      = '#D1D5DB',
    dark    = '#111827',
    t_rag   = '#0891B2',
    t_llm   = '#059669',
    t_fx    = '#EA580C',
    t_db    = '#7C3AED',
    panel   = '#F8FAFC',
    pbd     = '#CBD5E1',
)

FW, FH = 17, 19
fig = plt.figure(figsize=(FW, FH), facecolor=C['bg'])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis('off')


def box(ax, x, y, w, h, fc, text, fs=9.5, tc='white', ec=None, ls='-'):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0,rounding_size=0.1',
                          fc=fc, ec=ec or fc, lw=1.4, linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold',
            multialignment='center', zorder=4, linespacing=1.5)


def arr(ax, x1, y1, x2, y2, col='#475569', lw=1.6, rad=0.0,
        label='', lfs=8, lfc=None, lx=None, ly=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                mutation_scale=14), zorder=5)
    if label:
        mx = lx if lx is not None else (x1+x2)/2
        my = ly if ly is not None else (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=lfs, color=lfc or col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc=C['bg'],
                          ec='none', alpha=0.9), zorder=6)


def panel(ax, x, y, w, h, title='', tc=None):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0,rounding_size=0.12',
                       fc=C['panel'], ec=C['pbd'], lw=1.2, zorder=1)
    ax.add_patch(r)
    if title:
        ax.text(x + 0.2, y + h - 0.15, title, ha='left', va='top',
                fontsize=7.5, color=tc or C['gray'], fontweight='bold', zorder=2)


# ─── Title ─────────────────────────────────────────────────────────────
ax.text(FW/2, 18.55, 'TradeSimple — System Architecture',
        ha='center', fontsize=17, fontweight='bold', color=C['dark'])
ax.axhline(18.1, xmin=0.03, xmax=0.97, color=C['lt'], lw=1.0)


# ─── Infrastructure ────────────────────────────────────────────────────
panel(ax, 0.3, 16.85, 16.4, 1.1, 'INFRASTRUCTURE', tc=C['infra'])
box(ax, 0.55, 17.1, 3.6, 0.7, C['infra'], 'Streamlit Web UI  :8501')
arr(ax, 4.15, 17.45, 5.1, 17.45, C['infra'], lw=1.8, label='HTTP/SSE', lfs=7.5)
box(ax, 5.1, 17.1, 4.2, 0.7, C['infra'], 'FastAPI  :8000\n/analyze   /calculate')
arr(ax, 9.3, 17.45, 10.2, 17.45, C['infra'], lw=1.8)
box(ax, 10.2, 17.1, 6.2, 0.7, C['node'], 'LangGraph  StateGraph(AgentState)')


# ─── Main flow ─────────────────────────────────────────────────────────
NX, NW, NHT = 0.5, 7.5, 0.75
CX = NX + NW / 2   # 4.25

# START
ax.scatter([CX], [16.55], s=90, color=C['lt'], zorder=5)
arr(ax, CX, 16.55, CX, 16.0, C['lt'])

# 1. input_validator
box(ax, NX, 15.2, NW, NHT, C['node'],
    'input_validator\nLLM — extract fields  +  detect HS code in input')
arr(ax, CX, 16.0, CX, 15.95)

# error branch →
arr(ax, NX + NW, 15.575, 10.5, 15.575, C['err'], lw=1.5,
    label='missing fields', lfc=C['err'], lx=9.5, ly=15.74)
box(ax, 10.5, 15.2, 4.1, 0.75, C['err'], 'request_info\n→ END', fs=9)

# 2. supervisor
arr(ax, CX, 15.2, CX, 14.55)
box(ax, NX, 13.75, NW, NHT, C['node'], 'supervisor\nroute based on current state')

# ── SHORTCUT: HS code known ──────────────────────────────────────────
# Arrow from supervisor's right edge, curves right, lands on tax_calculator right edge
# supervisor right: (NX+NW, 14.125)  →  tax_calc right: (NX+NW, 10.125)
ax.annotate('', xy=(NX + NW, 10.125), xytext=(NX + NW, 14.125),
            arrowprops=dict(arrowstyle='->', color=C['skip'], lw=2.2,
                            connectionstyle='arc3,rad=-0.4',
                            linestyle='dashed', mutation_scale=14), zorder=5)
# shortcut label
ax.text(10.4, 12.3, 'HS code detected\n→  skip HITL\n(phase = complete\n/analyze only)',
        ha='center', va='center', fontsize=8.5,
        color=C['skip'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', fc='#F0FDF4',
                  ec=C['skip'], lw=1.3, alpha=0.97), zorder=6)

# ── HITL PATH ──────────────────────────────────────────────────────────
arr(ax, CX, 13.75, CX, 13.05,
    label='no HS code', lfc=C['gray'], lfs=7.5, lx=CX + 1.2, ly=13.4)

# 3. parallel_fetch
box(ax, NX, 12.25, NW, NHT, C['node'],
    'parallel_fetch  (asyncio.gather)\nHS candidates  +  exchange rate')

# → HITL box
arr(ax, NX + NW, 12.625, 12.1, 12.625, C['hitl'], lw=1.8,
    label='candidates', lfc=C['hitl'], lx=10.5, ly=12.8)

panel(ax, 12.1, 11.1, 4.6, 2.0)
hitl_hdr = FancyBboxPatch((12.1, 12.5), 4.6, 0.6,
                           boxstyle='round,pad=0,rounding_size=0.08',
                           fc=C['hitl'], ec='none', zorder=3)
ax.add_patch(hitl_hdr)
ax.text(14.4, 12.8, 'Human-in-the-Loop',
        ha='center', va='center', fontsize=10.5, color='white', fontweight='bold', zorder=4)
ax.text(14.4, 12.22, '3 HS code candidates', ha='center',
        fontsize=8.5, color='#92400E', fontweight='bold', zorder=4)
ax.text(14.4, 11.85, 'User selects the correct code', ha='center',
        fontsize=8, color='#92400E', zorder=4)
ax.text(14.4, 11.5,  'graph stops — POST /calculate', ha='center',
        fontsize=7.5, color=C['hitl'], style='italic', fontweight='bold', zorder=4)
ax.text(14.4, 11.22, 'to continue', ha='center',
        fontsize=7.5, color=C['hitl'], style='italic', fontweight='bold', zorder=4)

# HITL → tax_calculator (curved)
ax.annotate('', xy=(CX, 10.05), xytext=(12.1, 11.6),
            arrowprops=dict(arrowstyle='->', color=C['hitl'], lw=2.0,
                            connectionstyle='arc3,rad=0.4',
                            mutation_scale=14), zorder=5)
ax.text(8.5, 10.8, 'selection done',
        ha='center', va='center', fontsize=8.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc=C['hitl'], ec='none'), zorder=6)

# 4. tax_calculator
arr(ax, CX, 12.25, CX, 11.4)
box(ax, NX, 9.5, NW, NHT, C['node'],
    'tax_calculator\ntariff rate  +  exchange rate  +  VAT (10%)')

# 5. report_writer
arr(ax, CX, 9.5, CX, 8.8)
box(ax, NX, 8.0, NW, NHT, C['node'],
    'report_writer  (asyncio.gather)\nPDF  /  Word  /  Excel')

# END
arr(ax, CX, 8.0, CX, 7.4)
ax.scatter([CX], [7.35], s=130, color=C['dark'], zorder=5)
ax.scatter([CX], [7.35], s=55,  color=C['bg'],   zorder=6)
ax.text(CX + 0.5, 7.35, 'END', va='center', fontsize=9, color=C['gray'])


# ─── Tools ─────────────────────────────────────────────────────────────
panel(ax, 0.3, 0.2, 16.4, 3.3, 'TOOLS & EXTERNAL SERVICES', tc=C['gray'])

tools = [
    (0.55,  C['t_rag'], 'FAISS Dual-Index  (RAG)',
     ['Embedding: nlpai-lab/KURE-v1',
      'unified (PDF+CSV) + pdf-only',
      'PDF_QUOTA=3  |  Hit@5: 37%→63%']),
    (4.65,  C['t_llm'], 'LLM  (OpenAI)',
     ['GPT-4o-mini — input parsing / ReAct',
      'HS code search & classification',
      'GPT-4o — report generation']),
    (8.75,  C['t_fx'],  'Exchange Rate API',
     ['exchangerate-api.com',
      'Real-time FX  (1x per session)',
      'fetched in parallel_fetch node']),
    (12.85, C['t_db'],  'Tariff DB  (CSV)',
     ['tariff_by_hs.csv',
      'tariff rate · HS code · name',
      'pandas DataFrame lookup']),
]

for tx, color, title, lines in tools:
    TW = 3.8
    box(ax, tx, 2.8, TW, 0.5, color, title, fs=8.5)
    yt = 2.57
    for ln in lines:
        ax.text(tx + 0.15, yt, ln, va='top', fontsize=7.8, color=C['dark'], zorder=3)
        yt -= 0.46


plt.savefig(OUT, dpi=150, bbox_inches='tight',
            facecolor=C['bg'], edgecolor='none')
print(f'Saved: {os.path.abspath(OUT)}')
