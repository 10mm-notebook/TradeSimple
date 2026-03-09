"""
TradeSimple Architecture Diagram Generator
Run: python scripts/generate_architecture.py
Outputs: architecture.png at project root
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Constants ─────────────────────────────────────────────────────────
FIG_W, FIG_H = 22, 15
BG = '#F8F9FA'

C_INFRA  = '#2563EB'
C_NODE   = '#4F46E5'
C_HITL   = '#D97706'
C_TOOL   = '#059669'
C_RAG    = '#0891B2'
C_STATE  = '#7C3AED'
C_ARROW  = '#6B7280'
WHITE    = '#FFFFFF'
LAVENDER = '#EDE9FE'
CYAN_BG  = '#ECFEFF'
GREEN_BG = '#F0FDF4'
AMBER_BG = '#FFFBEB'
ORANGE_BG = '#FFF7ED'
VIOLET_BG = '#F5F3FF'


def box(ax, x, y, w, h, color, text,
        fontsize=9, text_color=WHITE,
        edgecolor=WHITE, lw=1.5, rounding=0.12, alpha=1.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=color, edgecolor=edgecolor, linewidth=lw,
        alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold',
            multialignment='center', zorder=4)


def arr(ax, x1, y1, x2, y2, color=C_ARROW,
        label='', loff=(0, 0.18), lw=1.8, rad=0.0):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=lw,
            connectionstyle=f'arc3,rad={rad}',
            mutation_scale=14,
        ),
        zorder=5,
    )
    if label:
        mx = (x1 + x2) / 2 + loff[0]
        my = (y1 + y2) / 2 + loff[1]
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=7.5, color=color, fontstyle='italic', zorder=6)


# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor(BG)

# ── Title ─────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, 14.5, 'TradeSimple — System Architecture',
        ha='center', va='center', fontsize=17,
        fontweight='bold', color='#111827')

# ══════════════════════════════════════════════════════════════════════
# ROW 1: Infrastructure
# ══════════════════════════════════════════════════════════════════════
ax.text(0.5, 13.65, 'Infrastructure', fontsize=8, color='#9CA3AF', va='center')

box(ax, 0.4, 12.5, 3.8, 1.0, C_INFRA, 'Streamlit Web UI\n:8501', fontsize=10)
arr(ax, 4.2, 13.0, 5.6, 13.0, color=C_INFRA, label='HTTP / SSE')
box(ax, 5.6, 12.5, 4.4, 1.0, C_INFRA,
    'FastAPI  :8000\n/analyze     /calculate', fontsize=9.5)
arr(ax, 10.0, 13.0, 11.3, 13.0, color=C_INFRA)
box(ax, 11.3, 12.5, 5.0, 1.0, C_NODE,
    'LangGraph  StateGraph(AgentState)', fontsize=10)

# ══════════════════════════════════════════════════════════════════════
# ROW 2: Graph Flow  (left/centre)
# ══════════════════════════════════════════════════════════════════════
ax.text(0.5, 12.1, 'LangGraph Flow', fontsize=8, color='#9CA3AF', va='center')

# START
ax.text(2.3, 11.65, 'START', ha='center', fontsize=9,
        color='#374151', style='italic')
arr(ax, 2.3, 11.48, 2.3, 10.85)

# input_validator
box(ax, 0.5, 10.0, 3.6, 0.82, C_NODE,
    'input_validator\n(LLM — parse & extract)', fontsize=8.5)

# missing info path
arr(ax, 1.3, 10.0, 1.3, 8.85,
    color='#EF4444', label='missing fields', loff=(0.65, 0.15))

# request_info
box(ax, 0.2, 8.05, 2.5, 0.75, '#6B7280',
    'request_info\n(ask for input)', fontsize=8)
ax.text(1.45, 7.72, '-> END', ha='center', fontsize=8, color='#9CA3AF')

# ok -> supervisor
arr(ax, 4.1, 10.41, 5.5, 10.41, color=C_ARROW)

# supervisor
box(ax, 5.5, 10.0, 2.8, 0.82, C_NODE, 'supervisor\n(router)', fontsize=8.5)

# supervisor -> parallel_fetch
arr(ax, 6.9, 10.0, 6.9, 8.85)

# parallel_fetch
box(ax, 4.8, 8.0, 4.2, 0.82, C_NODE,
    'parallel_fetch  (asyncio.gather)\nHS candidates  +  exchange rate', fontsize=8.5)

# parallel_fetch -> HITL
arr(ax, 9.0, 8.41, 10.1, 8.41, color=C_HITL, label='candidates')

# HITL box
hitl = FancyBboxPatch(
    (10.1, 7.45), 5.2, 1.35,
    boxstyle='round,pad=0.05,rounding_size=0.15',
    facecolor=AMBER_BG, edgecolor=C_HITL, linewidth=2.2, zorder=3,
)
ax.add_patch(hitl)
ax.text(12.7, 8.28, 'Human-in-the-Loop',
        ha='center', va='center', fontsize=10.5,
        fontweight='bold', color='#92400E', zorder=4)
ax.text(12.7, 7.82, '3 HS code candidates  ->  user selects',
        ha='center', va='center', fontsize=8.5, color='#92400E', zorder=4)

# HITL -> tax_calculator (curved re-entry)
arr(ax, 12.7, 7.45, 9.2, 5.85,
    color=C_HITL, rad=-0.25,
    label='selection done  (phase=tax_calculator)', loff=(1.2, 0.18))

# tax_calculator
box(ax, 6.7, 5.0, 3.5, 0.82, C_NODE,
    'tax_calculator\n(tariff + VAT)', fontsize=8.5)

arr(ax, 8.45, 5.0, 8.45, 3.85)

# report_writer
box(ax, 6.7, 3.0, 3.5, 0.82, C_NODE,
    'report_writer\n(PDF / Word / Excel  parallel)', fontsize=8.5)

arr(ax, 8.45, 3.0, 8.45, 2.38)
ax.text(8.45, 2.2, 'END', ha='center', fontsize=9,
        color='#374151', style='italic')

# supervisor re-entry arrows (dashed, gray)
ax.annotate('', xy=(7.5, 5.82), xytext=(8.3, 10.0),
            arrowprops=dict(arrowstyle='->', color='#D1D5DB', lw=1.2,
                            connectionstyle='arc3,rad=-0.38'),
            zorder=2)
ax.text(9.7, 8.2, 'supervisor\nre-entry', fontsize=7,
        color='#9CA3AF', ha='center', style='italic')

# ══════════════════════════════════════════════════════════════════════
# ROW 2 (right): AgentState panel
# ══════════════════════════════════════════════════════════════════════
state_bg = FancyBboxPatch(
    (16.0, 3.8), 5.7, 9.1,
    boxstyle='round,pad=0.05,rounding_size=0.15',
    facecolor=LAVENDER, edgecolor=C_STATE, linewidth=1.8,
    alpha=0.85, zorder=2,
)
ax.add_patch(state_bg)
ax.text(18.85, 12.65, 'AgentState',
        ha='center', va='center', fontsize=11,
        fontweight='bold', color=C_STATE)

groups = [
    ('User Input',
     'item_name  ·  quantity  ·  currency\n'
     'unit_price  ·  total_foreign_price\n'
     'raw_material  ·  processing_method\n'
     'product_form  ·  main_material'),
    ('HS Code',
     'hs_code  ·  hs_code_rationale\n'
     'hs_code_candidates  ·  tariff_rate'),
    ('Calculation',
     'exchange_rate  ·  tax_amount\n'
     'total_cost  ·  report_paths'),
    ('Flow Control',
     'messages  ·  current_phase\n'
     'missing_info  ·  report_id  ·  error'),
]

y_cur = 12.2
for title, content in groups:
    ax.text(16.3, y_cur, f'  {title}',
            fontsize=8.5, fontweight='bold', color='#4C1D95', va='top')
    ax.text(16.5, y_cur - 0.38, content,
            fontsize=7.8, color='#374151', va='top', family='monospace')
    y_cur -= 2.05

# ══════════════════════════════════════════════════════════════════════
# ROW 3: Tools & External Services  (bottom)
# ══════════════════════════════════════════════════════════════════════
ax.text(0.4, 1.85, 'Tools & External Services',
        fontsize=8.5, color='#6B7280')

# RAG
rag_bg = FancyBboxPatch((0.3, 0.25), 5.3, 1.48,
                         boxstyle='round,pad=0.05,rounding_size=0.12',
                         facecolor=CYAN_BG, edgecolor=C_RAG, linewidth=1.5, zorder=3)
ax.add_patch(rag_bg)
ax.text(2.95, 1.6, 'FAISS Dual-Index  (RAG)',
        ha='center', fontsize=9, fontweight='bold', color='#155E75', zorder=4)
ax.text(2.95, 1.25, 'Embedding: nlpai-lab/KURE-v1',
        ha='center', fontsize=8, color='#374151', zorder=4)
ax.text(2.95, 0.9,
        'unified (PDF+CSV)  +  pdf-only  |  PDF_QUOTA=3',
        ha='center', fontsize=7.5, color='#374151', family='monospace', zorder=4)
ax.text(2.95, 0.52, 'Hit@5 HS6:  37.1%  ->  62.9%  (+25.8%p)',
        ha='center', fontsize=8, color='#059669', fontweight='bold', zorder=4)

# LLM
llm_bg = FancyBboxPatch((5.9, 0.25), 4.0, 1.48,
                         boxstyle='round,pad=0.05,rounding_size=0.12',
                         facecolor=GREEN_BG, edgecolor=C_TOOL, linewidth=1.5, zorder=3)
ax.add_patch(llm_bg)
ax.text(7.9, 1.6, 'LLM  (OpenAI)',
        ha='center', fontsize=9, fontweight='bold', color='#065F46', zorder=4)
ax.text(7.9, 1.23, 'GPT-4o-mini  (input parsing / ReAct)',
        ha='center', fontsize=8, color='#374151', zorder=4)
ax.text(7.9, 0.88, 'HS code search & classification',
        ha='center', fontsize=8, color='#6B7280', zorder=4)
ax.text(7.9, 0.52, 'GPT-4o  (report generation)',
        ha='center', fontsize=7.5, color='#9CA3AF', zorder=4)

# Exchange Rate API
er_bg = FancyBboxPatch((10.2, 0.25), 4.2, 1.48,
                        boxstyle='round,pad=0.05,rounding_size=0.12',
                        facecolor=ORANGE_BG, edgecolor='#EA580C', linewidth=1.5, zorder=3)
ax.add_patch(er_bg)
ax.text(12.3, 1.6, 'Exchange Rate API',
        ha='center', fontsize=9, fontweight='bold', color='#7C2D12', zorder=4)
ax.text(12.3, 1.23, 'exchangerate-api.com',
        ha='center', fontsize=8, color='#374151', zorder=4)
ax.text(12.3, 0.88, 'Real-time FX  (1x per session)',
        ha='center', fontsize=8, color='#6B7280', zorder=4)
ax.text(12.3, 0.52, 'fetched in parallel_fetch node',
        ha='center', fontsize=7.5, color='#9CA3AF', zorder=4)

# Tariff DB
tariff_bg = FancyBboxPatch((14.7, 0.25), 4.0, 1.48,
                            boxstyle='round,pad=0.05,rounding_size=0.12',
                            facecolor=VIOLET_BG, edgecolor=C_STATE, linewidth=1.5, zorder=3)
ax.add_patch(tariff_bg)
ax.text(16.7, 1.6, 'Tariff DB  (CSV)',
        ha='center', fontsize=9, fontweight='bold', color='#4C1D95', zorder=4)
ax.text(16.7, 1.23, 'tariff_by_hs.csv',
        ha='center', fontsize=8, color='#374151', family='monospace', zorder=4)
ax.text(16.7, 0.88, 'tariff rate  ·  HS code  ·  item name',
        ha='center', fontsize=8, color='#6B7280', zorder=4)
ax.text(16.7, 0.52, 'pandas DataFrame lookup (no embedding)',
        ha='center', fontsize=7.5, color='#9CA3AF', zorder=4)

# ── Save ──────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), '..', 'architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print(f"Saved: {os.path.abspath(out_path)}")
