"""
TradeSimple Architecture Diagram — redesigned for readability
Run: python scripts/generate_architecture.py
Outputs: architecture.png at project root
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ─── Palette ─────────────────────────────────────────────────────────
P = dict(
    bg        = '#FAFBFC',
    panel     = '#EEF2F7',   # section background
    border    = '#C7D2E0',   # panel border
    shadow    = '#B0BCC8',   # drop shadow
    infra     = '#1D4ED8',   # blue  – infra
    infra_lt  = '#DBEAFE',
    node      = '#4338CA',   # indigo – graph nodes
    node_lt   = '#E0E7FF',
    hitl_bg   = '#FFFBEB',
    hitl_bd   = '#D97706',   # amber – HITL
    hitl_tx   = '#92400E',
    state_bg  = '#F5F3FF',
    state_bd  = '#7C3AED',   # violet – AgentState
    state_hd  = '#4C1D95',
    rag_bg    = '#ECFEFF',
    rag_bd    = '#0891B2',
    rag_tx    = '#155E75',
    llm_bg    = '#F0FDF4',
    llm_bd    = '#059669',
    llm_tx    = '#065F46',
    api_bg    = '#FFF7ED',
    api_bd    = '#EA580C',
    api_tx    = '#7C2D12',
    db_bg     = '#F5F3FF',
    db_bd     = '#7C3AED',
    db_tx     = '#4C1D95',
    err       = '#DC2626',
    gray      = '#6B7280',
    gray_lt   = '#9CA3AF',
    dark      = '#111827',
    arrow     = '#475569',
    mono      = '#1E293B',
    green     = '#16A34A',
)


# ─── Helpers ─────────────────────────────────────────────────────────
def shadow_box(ax, x, y, w, h, fc, ec, text,
               fontsize=10, tc='white', lw=1.6, rnd=0.08, bold=True):
    """Rounded box with drop shadow."""
    # shadow
    sh = FancyBboxPatch((x + 0.06, y - 0.06), w, h,
                        boxstyle=f'round,pad=0,rounding_size={rnd}',
                        fc=P['shadow'], ec='none', zorder=2)
    ax.add_patch(sh)
    # main
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={rnd}',
                          fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(rect)
    fw = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, color=tc, fontweight=fw,
            multialignment='center', zorder=4)


def node(ax, x, y, w, h, text, fontsize=9.5):
    """Graph node — indigo with white text."""
    shadow_box(ax, x, y, w, h, P['node'], P['node'], text,
               fontsize=fontsize, tc='white', rnd=0.09)


def tag(ax, x, y, label, color=None, fontsize=7.5):
    """Small pill-shaped section label."""
    c = color or P['gray']
    ax.text(x, y, f'  {label}  ',
            ha='left', va='center', fontsize=fontsize,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=c, ec='none'),
            zorder=5)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, rad=0.0,
          label='', label_color=None, label_side='right'):
    c = color or P['arrow']
    lc = label_color or c
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                mutation_scale=16),
                zorder=5)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = 0.18 if label_side == 'right' else -0.18
        ax.text(mx + dx, my, label,
                ha='left' if label_side == 'right' else 'right',
                va='center', fontsize=7.5, color=lc,
                fontstyle='italic', zorder=6)


def section_panel(ax, x, y, w, h, title=None, title_color=None):
    """Light background panel with optional title."""
    bg = FancyBboxPatch((x, y), w, h,
                        boxstyle='round,pad=0,rounding_size=0.1',
                        fc=P['panel'], ec=P['border'], lw=1.4,
                        zorder=1, alpha=1.0)
    ax.add_patch(bg)
    if title:
        tc = title_color or P['gray']
        ax.text(x + 0.25, y + h - 0.18, title,
                ha='left', va='top', fontsize=8,
                color=tc, fontweight='bold', zorder=2,
                bbox=dict(boxstyle='round,pad=0.2', fc=P['bg'],
                          ec='none', alpha=0.7))


# ─── Canvas ──────────────────────────────────────────────────────────
FW, FH = 22, 20
fig = plt.figure(figsize=(FW, FH), facecolor=P['bg'])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
ax.set_facecolor(P['bg'])

# ─── Title ───────────────────────────────────────────────────────────
ax.text(FW/2, 19.4, 'TradeSimple', ha='center', va='center',
        fontsize=22, fontweight='bold', color=P['dark'])
ax.text(FW/2, 18.85, 'System Architecture',
        ha='center', va='center', fontsize=13, color=P['gray'])
ax.axhline(18.55, xmin=0.04, xmax=0.96, color=P['border'], lw=1.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — Infrastructure  (y 16.8 – 18.3)
# ═══════════════════════════════════════════════════════════════════
section_panel(ax, 0.4, 16.8, 21.2, 1.5, title='INFRASTRUCTURE', title_color=P['infra'])

# Streamlit
shadow_box(ax, 0.8, 17.1, 4.2, 0.9, P['infra'], P['infra'],
           'Streamlit Web UI  :8501', fontsize=10.5)

# arrow HTTP/SSE
arrow(ax, 5.0, 17.55, 6.5, 17.55, color=P['infra'], lw=2.0)
ax.text(5.75, 17.72, 'HTTP / SSE', ha='center', fontsize=8,
        color=P['infra'], fontweight='bold')

# FastAPI
shadow_box(ax, 6.5, 17.1, 5.0, 0.9, P['infra'], P['infra'],
           'FastAPI  :8000\n/analyze   /calculate', fontsize=10)

# arrow
arrow(ax, 11.5, 17.55, 13.0, 17.55, color=P['infra'], lw=2.0)

# LangGraph
shadow_box(ax, 13.0, 17.1, 8.0, 0.9, P['node'], P['node'],
           'LangGraph  StateGraph(AgentState)', fontsize=10.5)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — Agent Graph Flow  (y 3.5 – 16.5)
# ═══════════════════════════════════════════════════════════════════
section_panel(ax, 0.4, 3.5, 13.5, 13.0, title='LANGGRAPH AGENT FLOW', title_color=P['node'])

# ---  Main vertical flow (x center ~5)  ---
NW = 5.8   # node width
NX = 1.0   # node left x
CX = NX + NW/2  # node center x

# START marker
ax.annotate('', xy=(CX, 16.3), xytext=(CX, 16.6),
            arrowprops=dict(arrowstyle='->', color=P['gray_lt'], lw=1.5,
                            mutation_scale=12), zorder=5)
ax.scatter([CX], [16.65], s=120, color=P['gray_lt'], zorder=5)
ax.text(CX + 0.35, 16.65, 'START', va='center', fontsize=8.5,
        color=P['gray_lt'], style='italic')

# input_validator
node(ax, NX, 15.3, NW, 0.9,
     'input_validator\nLLM — parse & extract input')
arrow(ax, CX, 16.3, CX, 16.2)

# error path (right)
arrow(ax, NX + NW, 15.75, 9.0, 15.75, color=P['err'], lw=1.6,
      label='missing fields', label_color=P['err'])
shadow_box(ax, 9.0, 15.3, 3.8, 0.9,
           P['gray'], P['gray'], 'request_info\nask for input',
           fontsize=9, tc='white', rnd=0.08)
ax.text(11.3, 15.07, 'END', ha='center', fontsize=8,
        color=P['gray_lt'], style='italic')

# ok path (down)
arrow(ax, CX, 15.3, CX, 14.35, label='all fields OK', label_color=P['green'])

# supervisor
node(ax, NX, 13.4, NW, 0.9, 'supervisor\nrouter — decide next node')
arrow(ax, CX, 14.3, CX, 14.2)

arrow(ax, CX, 13.4, CX, 12.45)

# parallel_fetch
node(ax, NX, 11.5, NW, 0.9,
     'parallel_fetch  (asyncio.gather)\nHS candidates  +  exchange rate')
arrow(ax, CX, 12.4, CX, 12.3)

# candidates arrow (right to HITL)
arrow(ax, NX + NW, 11.95, 14.9, 11.95,
      color=P['hitl_bd'], lw=2.0, label='candidates', label_color=P['hitl_bd'])

# HITL box (right panel)
hitl = FancyBboxPatch((14.9, 10.7), 6.7, 2.0,
                      boxstyle='round,pad=0,rounding_size=0.12',
                      fc=P['hitl_bg'], ec=P['hitl_bd'], lw=2.5, zorder=3)
ax.add_patch(hitl)
# HITL header strip
hdr = FancyBboxPatch((14.9, 12.1), 6.7, 0.6,
                     boxstyle='round,pad=0,rounding_size=0.10',
                     fc=P['hitl_bd'], ec='none', zorder=4)
ax.add_patch(hdr)
ax.text(18.25, 12.4, 'Human-in-the-Loop',
        ha='center', va='center', fontsize=11.5,
        color='white', fontweight='bold', zorder=5)
ax.text(18.25, 11.55, '3 HS code candidates', ha='center',
        fontsize=9.5, color=P['hitl_tx'], fontweight='bold', zorder=5)
ax.text(18.25, 11.15, 'User reviews and selects the correct HS code',
        ha='center', fontsize=8.5, color=P['hitl_tx'], zorder=5)
ax.text(18.25, 10.82, 'Each candidate includes classification rationale',
        ha='center', fontsize=8, color='#B45309', zorder=5)

# HITL selection -> back to flow (curved arrow going left-down)
ax.annotate('', xy=(CX, 10.3), xytext=(14.9, 11.0),
            arrowprops=dict(arrowstyle='->', color=P['hitl_bd'], lw=2.5,
                            connectionstyle='arc3,rad=0.3',
                            mutation_scale=18), zorder=5)
# label box for return arrow
ax.text(10.5, 10.8, 'selection done',
        ha='center', fontsize=9, color='white', fontweight='bold', zorder=7,
        bbox=dict(boxstyle='round,pad=0.3', fc=P['hitl_bd'], ec='none', alpha=0.9))

# tax_calculator
node(ax, NX, 9.4, NW, 0.9, 'tax_calculator\ntariff rate  +  VAT  (10%)')
arrow(ax, CX, 10.4, CX, 10.3)

arrow(ax, CX, 9.4, CX, 8.45)

# report_writer
node(ax, NX, 7.5, NW, 0.9,
     'report_writer  (asyncio.gather)\nPDF  /  Word  /  Excel')
arrow(ax, CX, 8.4, CX, 8.3)

arrow(ax, CX, 7.5, CX, 6.8)
ax.scatter([CX], [6.72], s=120, color=P['gray_lt'], zorder=5)
ax.text(CX + 0.35, 6.72, 'END', va='center', fontsize=8.5,
        color=P['gray_lt'], style='italic')

# supervisor re-entry note (subtle label only, no confusing arrow)
ax.text(8.2, 12.9, 'loops back to supervisor', ha='center',
        fontsize=7.5, color=P['gray_lt'], style='italic', zorder=3)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — AgentState  (y 9.8 – 16.5, right panel x 14.9)
# ═══════════════════════════════════════════════════════════════════
section_panel(ax, 14.9, 3.5, 6.7, 7.1, title='AGENTSTATE', title_color=P['state_hd'])

AX, AW = 15.1, 6.3  # agentstate panel x, width

groups = [
    ('User Input',    P['infra'],
     ['item_name   quantity   currency',
      'unit_price   total_foreign_price',
      'raw_material   processing_method',
      'product_form   main_material']),
    ('HS Code',       P['node'],
     ['hs_code   hs_code_rationale',
      'hs_code_candidates   tariff_rate']),
    ('Calculation',   '#0891B2',
     ['exchange_rate   tax_amount',
      'total_cost   report_paths']),
    ('Flow Control',  P['gray'],
     ['messages   current_phase',
      'missing_info   report_id   error']),
]

y_g = 10.0
for title, color, lines in groups:
    # mini header
    mini_h = FancyBboxPatch((AX, y_g - 0.05), AW, 0.38,
                            boxstyle='round,pad=0,rounding_size=0.05',
                            fc=color, ec='none', zorder=3, alpha=0.85)
    ax.add_patch(mini_h)
    ax.text(AX + AW/2, y_g + 0.14, title, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold', zorder=4)
    y_g -= 0.42
    for line in lines:
        ax.text(AX + 0.25, y_g, line, va='top',
                fontsize=7.8, color=P['mono'],
                family='monospace', zorder=3)
        y_g -= 0.38
    y_g -= 0.18


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — Tools & External Services  (y 0.2 – 3.3)
# ═══════════════════════════════════════════════════════════════════
section_panel(ax, 0.4, 0.2, 21.2, 3.1,
              title='TOOLS & EXTERNAL SERVICES', title_color=P['gray'])

tools = [
    # (x, bg, border, title_color, title, lines)
    (0.7,  P['rag_bg'],  P['rag_bd'],  P['rag_tx'],
     'FAISS Dual-Index  (RAG)',
     ['Embedding: nlpai-lab/KURE-v1',
      'unified (PDF+CSV)  +  pdf-only',
      'PDF_QUOTA = 3  |  Hit@5: 37.1% -> 62.9%']),
    (5.9,  P['llm_bg'],  P['llm_bd'],  P['llm_tx'],
     'LLM  (OpenAI)',
     ['GPT-4o-mini — input parsing / ReAct',
      'HS code search & classification',
      'GPT-4o — report generation']),
    (11.1, P['api_bg'],  P['api_bd'],  P['api_tx'],
     'Exchange Rate API',
     ['exchangerate-api.com',
      'Real-time FX  (1x per session)',
      'fetched in parallel_fetch node']),
    (16.3, P['db_bg'],   P['db_bd'],   P['db_tx'],
     'Tariff DB  (CSV)',
     ['tariff_by_hs.csv',
      'tariff rate  ·  HS code  ·  item name',
      'pandas DataFrame lookup']),
]

for (tx, bg, bd, tc, title, lines) in tools:
    TW = 4.9
    # card shadow
    sh = FancyBboxPatch((tx + 0.06, 0.44), TW, 2.45,
                        boxstyle='round,pad=0,rounding_size=0.1',
                        fc=P['shadow'], ec='none', zorder=2)
    ax.add_patch(sh)
    # card bg
    card = FancyBboxPatch((tx, 0.5), TW, 2.45,
                          boxstyle='round,pad=0,rounding_size=0.1',
                          fc=bg, ec=bd, lw=1.5, zorder=3)
    ax.add_patch(card)
    # title strip
    strip = FancyBboxPatch((tx, 2.35), TW, 0.6,
                           boxstyle='round,pad=0,rounding_size=0.08',
                           fc=bd, ec='none', zorder=4)
    ax.add_patch(strip)
    ax.text(tx + TW/2, 2.65, title,
            ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', zorder=5)
    # body lines
    y_t = 2.22
    for ln in lines:
        ax.text(tx + 0.2, y_t, ln, va='top',
                fontsize=8, color=P['dark'], zorder=5)
        y_t -= 0.52


# ─── Save ────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), '..', 'architecture.png')
plt.savefig(out_path, dpi=160, bbox_inches='tight',
            facecolor=P['bg'], edgecolor='none')
print(f'Saved: {os.path.abspath(out_path)}')
