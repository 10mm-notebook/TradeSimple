"""
TradeSimple Architecture Diagram — Mermaid + Pyppeteer
Run: python scripts/generate_architecture.py
Requires: pip install pyppeteer
"""
import asyncio
import os
import tempfile
import pathlib

from pyppeteer import launch

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "architecture.png"

# ── Mermaid diagram ────────────────────────────────────────────────────
MERMAID = """
flowchart LR
    %% ── Left: main flow ──────────────────────────────────────────────
    subgraph FLOW ["  Agent Workflow"]
        direction TB
        subgraph INFRA ["🏗  Infrastructure"]
            direction LR
            UI["Streamlit\\n:8501"] -->|"HTTP / SSE"| API["FastAPI  :8000\\n/analyze  /calculate"] --> LG["LangGraph\\nStateGraph"]
        end

        INFRA --> ST(["● START"])
        ST --> IV["input_validator\\nLLM — extract fields + detect HS code"]

        IV -->|"❌  missing fields"| RI["request_info\\nask for input"] --> E1(["END"])
        IV -->|"✅  all fields OK"| SV["supervisor\\nroute based on state"]

        SV -->|"no HS code"| PF["parallel_fetch  (asyncio.gather)\\nHS candidates  +  exchange rate"]
        SV -->|"HS code detected  →  skip HITL"| TC

        PF --> HITL{{"🧑  Human-in-the-Loop\\n3 HS code candidates\\nUser reviews & selects\\n─────────────────────\\ngraph interrupt\\nPOST /calculate to continue"}}
        HITL -->|"selection done"| TC["tax_calculator\\ntariff rate  +  exchange rate  +  VAT (10%)"]

        TC --> RW["report_writer  (asyncio.gather)\\nPDF  /  Word  /  Excel"]
        RW --> E2(["● END"])
    end

    %% ── Right: tools ─────────────────────────────────────────────────
    subgraph TOOLS ["🛠  Tools & External Services"]
        direction TB
        RAG["FAISS Dual-Index  (RAG)\\nnlpai-lab/KURE-v1\\nHit@5: 37% → 63%"]
        LLM["LLM  (OpenAI)\\nGPT-4o-mini  /  GPT-4o"]
        FX["Exchange Rate API\\nexchangerate-api.com"]
        DB["Tariff DB  (CSV)\\ntariff_by_hs.csv  ·  pandas"]
        LS["LangSmith\\nTrace  ·  session_id = thread_id"]
        RAG ~~~ LLM ~~~ FX ~~~ DB ~~~ LS
    end

    FLOW ~~~ TOOLS

    %% ── Styles ───────────────────────────────────────────────────────
    classDef infra    fill:#1D4ED8,stroke:#1D4ED8,color:#fff,font-weight:bold
    classDef node     fill:#4338CA,stroke:#4338CA,color:#fff,font-weight:bold
    classDef hitl     fill:#FFFBEB,stroke:#D97706,stroke-width:2.5px,color:#92400E,font-weight:bold
    classDef err      fill:#FEF2F2,stroke:#DC2626,stroke-width:1.5px,color:#DC2626,font-weight:bold
    classDef tool     fill:#F8FAFC,stroke:#94A3B8,color:#1E293B
    classDef terminal fill:#374151,stroke:#374151,color:#fff

    class UI,API,LG infra
    class IV,SV,PF,TC,RW node
    class HITL hitl
    class RI err
    class RAG,LLM,FX,DB,LS tool
    class ST,E1,E2 terminal
"""

# ── HTML template ──────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body  {{ margin: 0; padding: 24px; background: #ffffff; font-family: sans-serif; }}
  .wrap {{ display: inline-block; }}
</style>
</head>
<body>
<div class="wrap">
  <pre class="mermaid">
{diagram}
  </pre>
</div>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'default',
    flowchart: {{ curve: 'basis', padding: 24, nodeSpacing: 50, rankSpacing: 80 }},
    themeVariables: {{
      fontSize: '18px',
      fontFamily: 'Segoe UI, Arial, sans-serif'
    }}
  }});
</script>
</body>
</html>
""".format(diagram=MERMAID)


async def render():
    # Write HTML to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False,
                                     mode='w', encoding='utf-8')
    tmp.write(HTML)
    tmp.close()

    # Use system-installed Chrome (pyppeteer's built-in Chromium download is unreliable)
    chrome_paths = [
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ]
    exe = next((p for p in chrome_paths if os.path.exists(p)), None)

    browser = await launch(
        headless=True,
        executablePath=exe,
        args=['--no-sandbox', '--disable-setuid-sandbox'],
    )
    page = await browser.newPage()
    await page.setViewport({'width': 2400, 'height': 1600, 'deviceScaleFactor': 3})
    await page.goto(f'file:///{tmp.name.replace(os.sep, "/")}',
                    waitUntil='networkidle0')

    # Wait for Mermaid SVG to render
    await page.waitForSelector('.mermaid svg', timeout=15000)
    await asyncio.sleep(0.5)   # let any animation finish

    # Screenshot just the diagram element
    element = await page.querySelector('.wrap')
    if element is None:
        element = await page.querySelector('body')
    await element.screenshot({'path': str(OUT)})

    await browser.close()
    os.unlink(tmp.name)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    asyncio.run(render())
