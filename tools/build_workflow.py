"""產生 workflow.html：工作流程、決策理由與實驗紀錄（給沒參與討論的同事讀）。

資料來源（筆電端；由 ws rsync 回來）：
  ~/runs/prewar_stem_tagging/alm/alm_20261115_1348/alm.parquet     AF-Next 描述＋自抽（條件 a、b）
  ~/runs/prewar_stem_tagging/alm/audex_20261116_1417/alm.parquet   Audex-2B 描述＋自抽（條件 a）
  ~/runs/prewar_stem_tagging/alm/local_llm_20261116/out_*.parquet  本機文字 LLM 重新抽取 AF-Next 描述
  ~/runs/prewar_stem_tagging/full/<run>/manifest_site.json          （選用）各窗長全量的彙整數字
樣式沿用 index.html 的 <style>。
用法：python tools/build_workflow.py
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

SITE = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "runs/prewar_stem_tagging"
ALM = RUNS / "alm"
CLASSES = ["pure_instrument", "pure_vocal", "pure_noise", "silence", "mixed"]
ZH = {"pure_instrument": "純樂器", "pure_vocal": "純人聲", "pure_noise": "純雜音", "silence": "靜音", "mixed": "混合"}
COLOR = {"pure_instrument": "var(--c-inst)", "pure_vocal": "var(--c-vocal)", "pure_noise": "var(--c-noise)",
         "silence": "var(--c-sil)", "mixed": "var(--c-mixed)"}
LLM_NAMES = {"openai/gpt-oss-120b": "gpt-oss-120b", "gemma-4-31b-it": "gemma-4-31b-it"}


def pct(x):
    return f"{100 * x:.1f}%"


def esc(s):
    return html.escape(str(s))


# ───────────────────────── 資料
af = pd.read_parquet(ALM / "alm_20261115_1348/alm.parquet")
ax = pd.read_parquet(ALM / "audex_20261116_1417/alm.parquet")
llm = {LLM_NAMES.get(pd.read_parquet(f).model.iloc[0], f.stem): pd.read_parquet(f)
       for f in sorted((ALM / "local_llm_20261116").glob("out_*.parquet"))}

sources = {  # 描述來源：同一批 305 段
    "AF-Next（不帶 metadata）": af[af.condition == "a"],
    "AF-Next（帶 metadata）": af[af.condition == "b"],
    "Audex-2B（不帶 metadata）": ax,
}
rule = af[af.condition == "a"][["seg_id", "stem", "final"]]
noise_ids = set(rule[rule.final == "pure_noise"].seg_id)
n_noise = len(noise_ids)


def stats(g):
    """類別有值即納入比較（AF-Next 的解析失敗多半只是 leak 欄為 null，類別仍有效）。"""
    v = g[g.cls.notna()]
    nz = g[g.seg_id.isin(noise_ids)]
    return dict(n=len(g), strict=int(g.json_strict.sum()), parse_ok=int(g.parse_ok.sum()), cls_ok=len(v),
                rule_agree=(v.cls == v.final).mean(), noise_hit=int((nz.cls == "pure_noise").sum()),
                sec=(g.t_describe + g.t_extract).mean())


# ───────────────────────── SVG：各來源 × stem 的類別比例
def stacked(rows, aria):
    """rows: [(label, sublabel, {cls: share}, n)]"""
    W, L, R, H, GAP = 734, 230, 24, 22, 10
    span = W - L - R
    h = len(rows) * (H + GAP) + 18
    out = [f'<svg class="viz" viewBox="0 0 {W} {h}" role="img" aria-label="{esc(aria)}">']
    for t in (0, .25, .5, .75, 1):
        x = L + span * t
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{h - 18}"/>'
                   f'<text class="tick" x="{x:.1f}" y="{h - 4}" text-anchor="middle">{int(t * 100)}%</text>')
    for i, (lab, sub, shares, n) in enumerate(rows):
        y = i * (H + GAP)
        out.append(f'<text class="lab" x="{L - 8}" y="{y + 15.4}" text-anchor="end">{esc(lab)}'
                   f'<tspan class="sub"> {esc(sub)}</tspan></text>')
        x = L
        for c in CLASSES:
            s = shares.get(c, 0)
            if s <= 0:
                continue
            w = span * s
            sil = ' sil' if c == "silence" else ''
            out.append(f'<rect class="seg{sil}" x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" height="{H}" rx="4" '
                       f'style="fill:{COLOR[c]}" tabindex="0" data-tip="{esc(lab)}｜{esc(sub)}｜{ZH[c]}：{pct(s)}（{round(s * n)} / {n} 段）"/>')
            if w > 34 and c not in ("silence", "mixed"):
                out.append(f'<text class="inlab" x="{x + 6:.1f}" y="{y + 15.4}">{round(s * 100)}%</text>')
            x += w
    out.append("</svg>")
    return "\n".join(out)


def shares(series):
    v = series.dropna()
    return v.value_counts(normalize=True).to_dict(), len(v)


dist_rows = []
for stem, sname in (("no_vocals", "伴奏軌"), ("vocals", "人聲軌")):
    r = rule[rule.stem == stem].final
    s, n = shares(r)
    dist_rows.append(("規則（AST＋CLAP）", sname, s, n))
    for name, g in sources.items():
        s, n = shares(g[g.stem == stem].cls)
        dist_rows.append((name, sname, s, n))
dist_svg = stacked(dist_rows, "各方法在 305 段上的類別比例")

legend = '<p class="legend">' + "".join(
    f'<span class="key"><i class="{"sil" if c == "silence" else ""}" style="background:{COLOR[c]}"></i>{ZH[c]}</span>'
    for c in CLASSES) + "</p>"

# ───────────────────────── 表：描述來源
src_rows = []
for name, g in sources.items():
    s = stats(g)
    src_rows.append(f"<tr><td>{esc(name)}</td><td class=num>{s['strict']}／{s['n']}</td><td class=num>{s['parse_ok']}／{s['n']}</td>"
                    f"<td class=num>{pct(s['rule_agree'])}</td><td class=num>{s['noise_hit']}／{n_noise}</td>"
                    f"<td class=num>{s['sec']:.1f} 秒</td></tr>")
src_table = ("<div class=tablewrap><table><thead><tr><th>描述與抽取</th><th class=num>嚴格 JSON</th><th class=num>欄位全部有效</th>"
             "<th class=num>類別與規則一致</th><th class=num>規則純雜音段<br>判為純雜音</th><th class=num>每段耗時</th></tr></thead><tbody>"
             + "".join(src_rows) + "</tbody></table></div>")

# ───────────────────────── 表：抽取模型（同一批 AF-Next 描述）


def verbatim(ev, desc):
    return isinstance(ev, str) and bool(ev.strip()) and ev.strip().strip(".").lower() in desc.lower()


ex_rows = []
for cond, cname in (("a", "不帶 metadata"), ("b", "帶 metadata")):
    g = af[af.condition == cond]
    ok = g[g.cls.notna()]
    ev = g.apply(lambda r: verbatim(r.evidence, r.description), axis=1).mean()
    nz = g[g.seg_id.isin(noise_ids)]
    ex_rows.append((cname, "AF-Next 自己抽取", f"{int(g.parse_ok.sum())}／{len(g)}", "（自身）",
                    pct((ok.cls == ok.final).mean()), pct(ev), f"{int((nz.cls == 'pure_noise').sum())}／{n_noise}",
                    f"{g.t_extract.mean():.1f} 秒"))
    for name, d in llm.items():
        h = d[d.condition == cond]
        both = h[h.afnext_cls.notna()]
        ev = h.apply(lambda r: verbatim(r.llm_evidence, r.description), axis=1).mean()
        nz = h[h.seg_id.isin(noise_ids)]
        ex_rows.append((cname, name, f"{int(h.parse_ok_y.sum())}／{len(h)}", pct((both.llm_cls == both.afnext_cls).mean()),
                        pct((h.llm_cls == h.rule_final).mean()), pct(ev), f"{int((nz.llm_cls == 'pure_noise').sum())}／{n_noise}",
                        f"{d.attrs.get('sec', float('nan')):.1f} 秒" if d.attrs.get('sec') else "約 1 秒"))
ex_table = ("<div class=tablewrap><table><thead><tr><th>描述條件</th><th>抽取模型</th><th class=num>JSON 有效</th>"
            "<th class=num>類別與 AF-Next 自抽一致</th><th class=num>類別與規則一致</th><th class=num>evidence 逐字出自描述</th>"
            "<th class=num>規則純雜音段<br>判為純雜音</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{a}</td><td>{b}</td><td class=num>{c}</td><td class=num>{d}</td><td class=num>{e}</td>"
                      f"<td class=num>{f}</td><td class=num>{g}</td></tr>" for a, b, c, d, e, f, g, _ in ex_rows)
            + "</tbody></table></div>")

# 抽取模型之間的主要差異：AF-Next 判 pure_vocal、本機 LLM 判 mixed
shift = []
for name, d in llm.items():
    for cond in ("a", "b"):
        h = d[d.condition == cond]
        shift.append(f"{name}（{'不帶' if cond == 'a' else '帶'} metadata）{int(((h.afnext_cls == 'pure_vocal') & (h.llm_cls == 'mixed')).sum())} 段")

# 兩個本機模型彼此的一致率（類別、殘留、雜音程度）
pair_html = ""
if len(llm) == 2:
    (n1, d1), (n2, d2) = llm.items()
    x = d1.merge(d2, on=["seg_id", "condition"], suffixes=("_1", "_2"))
    parts = []
    for cond, cname in (("a", "不帶"), ("b", "帶")):
        y = x[x.condition == cond]
        parts.append(f"{cname} metadata：類別 {pct((y.llm_cls_1 == y.llm_cls_2).mean())}、殘留 {pct((y.llm_leak_1 == y.llm_leak_2).mean())}、"
                     f"雜音程度 {pct((y.llm_noise_1 == y.llm_noise_2).mean())}")
    pair_html = (f"<li><b>兩個本機模型彼此一致</b>（{n1} 對 {n2}）：{'；'.join(parts)}。類別幾乎一樣，殘留與雜音程度分歧較大，"
                 "因為描述很少明說這兩件事，模型只能推測。這兩欄目前不宜直接採用。</li>"
                 "<li><b>速度</b>：gpt-oss-120b 610 筆約 13 分鐘（每筆約 1.2 秒）；gemma-4-31b-it 約 80 分鐘（每筆約 8 秒）。"
                 "gpt-oss 是混合專家模型，每個 token 只動用約 5B 參數；gemma 是 31B 稠密模型，所以慢得多。結果相近時選 gpt-oss。</li>")

# ───────────────────────── 例子


def desc_of(df, seg, cond=None):
    g = df[df.seg_id == seg]
    if cond:
        g = g[g.condition == cond]
    return g.description.iloc[0] if len(g) else ""


EX_SEG = "pilot_20261113_1903/4399"
ex_meta = af[(af.seg_id == EX_SEG) & (af.condition == "b")].context.iloc[0]
ax_examples = ax[ax.seg_id.isin(noise_ids)].description.drop_duplicates().head(40)
ax_pick = [d for d in ax_examples if re.search(r"spray|cigarette|gravel|Thank you|frying|whip", d)][:5]

# ───────────────────────── 窗長：三份 manifest_site.json（ws 經 Exchange 傳來）＋ data/manifest.json 的交集
MANI = Path.home() / "Exchange/files/prewar_stem_manifests"
WS = (4, 8, 10)
win = {}
for w, pat in ((4, "full_nv_w4_*"), (8, "full_nv_2*"), (10, "full_nv_w10_*")):
    hits = sorted(MANI.glob(f"{pat}.manifest_site.json"))
    win[w] = json.loads(hits[-1].read_text()) if hits else None
multi = json.loads((SITE / "data/manifest.json").read_text()).get("multiwindow")
if not (all(win.values()) and multi):
    raise SystemExit(f"缺窗長 manifest（{MANI}）或 data/manifest.json 的 multiwindow 區塊")


def agree(a, b):
    return win[a]["label_agreement"][f"vs_w{b}"]


win_svg = stacked([(f"{w} 秒窗", f"步距 {win[w]['hop_s']:g} 秒", win[w]["class_share"], win[w]["n_windows"]) for w in WS],
                  "三種窗長全量的類別比例（伴奏軌）")
win_table = ("<div class=tablewrap><table><thead><tr><th>窗長</th><th class=num>窗數</th><th class=num>純樂器</th>"
             "<th class=num>純雜音</th><th class=num>混合比例</th></tr></thead><tbody>" + "".join(
                 f"<tr><td>{w} 秒</td><td class=num>{win[w]['n_windows']:,}</td>"
                 f"<td class=num>{win[w]['hours']['pure_instrument']:.1f} h（{win[w]['n_intervals']['pure_instrument']:,} 段）</td>"
                 f"<td class=num>{win[w]['hours']['pure_noise']:.1f} h（{win[w]['n_intervals']['pure_noise']:,} 段）</td>"
                 f"<td class=num>{pct(win[w]['class_share']['mixed'])}</td></tr>" for w in WS) + "</tbody></table></div>")

# 逐秒一致率矩陣（對稱，取 overall）
mat_rows = []
for a in WS:
    cells = "".join("<td class=num>—</td>" if a == b else f"<td class=num>{pct(agree(a, b)['overall'])}</td>" for b in WS)
    mat_rows.append(f"<tr><th>{a} 秒</th>{cells}</tr>")
win_matrix = ("<div class=tablewrap><table><thead><tr><th>逐秒一致率</th>" + "".join(f"<th class=num>{b} 秒</th>" for b in WS)
              + "</tr></thead><tbody>" + "".join(mat_rows) + "</tbody></table></div>")

# 有方向的條件一致率：A 判為該類的點，在 B 仍判同類
PAIRS = [(4, 8), (8, 4), (8, 10), (10, 8), (4, 10), (10, 4)]
cond_rows = "".join(
    f"<tr><td>{a} 秒 → {b} 秒</td>" + "".join(f"<td class=num>{pct(agree(a, b)['same_given_this'][c])}</td>"
                                              for c in ("pure_instrument", "pure_noise", "mixed")) + "</tr>"
    for a, b in PAIRS)
win_cond = ("<div class=tablewrap><table><thead><tr><th>A 判為該類 → B 仍判同類</th><th class=num>純樂器</th>"
            "<th class=num>純雜音</th><th class=num>混合</th></tr></thead><tbody>" + cond_rows + "</tbody></table></div>")

si = multi["strict_all_three_hours"]
h8 = win[8]["hours"]
win_inter = ("<div class=tablewrap><table><thead><tr><th>取法</th><th class=num>純樂器</th><th class=num>純雜音</th></tr></thead><tbody>"
             f"<tr><td>只用 8 秒窗（目前的 segments_pure）</td><td class=num>{h8['pure_instrument']:.2f} h</td><td class=num>{h8['pure_noise']:.2f} h</td></tr>"
             f"<tr><td>三種窗長交集（嚴格）</td><td class=num>{si['pure_instrument']:.2f} h（{pct(si['pure_instrument'] / h8['pure_instrument'])}）</td>"
             f"<td class=num>{si['pure_noise']:.2f} h（{pct(si['pure_noise'] / h8['pure_noise'])}）</td></tr>"
             + (f"<tr><td>三種窗長交集（各取中心最近的窗；下載頁範例程式的算法）</td>"
                f"<td class=num>{sn['pure_instrument']:.2f} h（{pct(sn['pure_instrument'] / h8['pure_instrument'])}）</td>"
                f"<td class=num>{sn['pure_noise']:.2f} h（{pct(sn['pure_noise'] / h8['pure_noise'])}）</td></tr>"
                if (sn := multi.get("nearest_all_three_hours")) else "")
             + "</tbody></table></div>")

c48, c84 = agree(4, 8)["same_given_this"], agree(8, 4)["same_given_this"]
win_html = f"""{legend}
{win_svg}
{win_table}
<p>三種窗長都跑完全量（1,801 首），規則與模型完全相同，只換窗長與步距（步距皆為窗長的一半）。類別比例很接近；窗越短，判為混合的比例越高。</p>
<h3>逐秒比較</h3>
<p>每首每 1 秒取一點（t＝0.5、1.5…），各窗長取中心最近的窗之類別，只比三種窗長都涵蓋的點（{agree(4, 8)['n_points']:,} 點）。</p>
{win_matrix}
<p>一致率只說明彼此接近，<b>不說明哪個對</b>；哪個窗長精確率最高，要等黃金標準。下表是有方向的比較：A 判為某類的點，在 B 仍判同類的比例。</p>
{win_cond}
<ul>
<li><b>短窗的純樂器比較嚴格</b>：4 秒判純樂器的點，8 秒仍判純樂器的有 {pct(c48['pure_instrument'])}；反過來只有 {pct(c84['pure_instrument'])}。
4 秒的純樂器大致落在 8 秒純樂器的範圍內，多出來的部分在 4 秒被判為混合；推測是短暫的人聲殘留在長窗裡被平均掉，尚未抽聽確認。</li>
<li><b>純雜音最不穩定</b>：各方向只有 {pct(min(agree(a, b)['same_given_this']['pure_noise'] for a, b in PAIRS))} 到 {pct(max(agree(a, b)['same_given_this']['pure_noise'] for a, b in PAIRS))}。
純雜音的條件是三種分數都要很低，窗長一變，窗內只要混進一點音樂就不成立。</li>
<li><b>8 與 10 秒最接近</b>（{pct(agree(8, 10)['overall'])}），4 秒與兩者差得較多。</li>
</ul>
<h3>三種窗長的交集</h3>
<p>{esc(multi['strict_rule'])}</p>
{win_inter}
<p>嚴格交集讓純樂器少了約四成、純雜音少了約六成。交集是否真的比較準，同樣要等黃金標準；在那之前，三種窗長的逐窗類別都公開在下載區（<code>windows_class_w4／w8／w10.parquet</code>），使用者可以自行決定取 8 秒版或取交集。</p>"""

# ───────────────────────── 流程圖（SVG）


def flow_svg():
    W, BX, BW, H = 734, 16, 380, 40
    SX, SW = 452, 266
    main = [
        ("唱片數位檔：1,801 首、約 125 小時", None),
        ("demucs htdemucs 分離：伴奏軌、人聲軌", "人聲軌只計算音量，不跑模型"),
        ("切窗：8 秒、步距 4 秒（110,329 窗）", None),
        ("音量 RMS < −60 dB？", "是 → 靜音"),
        ("AST（16 kHz）＋CLAP（48 kHz）打分", "各存前 10 名 → windows_top10"),
        ("規則：兩個模型都同意才給純類別", "邊界案例 → 音訊語言模型審查（實驗一、二）"),
        ("類別：純樂器／純雜音／混合", "黃金標準：人工試聽 240 窗"),
        ("同一首相鄰同類窗合併 → segments_pure", None),
    ]
    gap = 30
    h = len(main) * (H + gap) + 4
    o = [f'<svg class="viz flow" viewBox="0 0 {W} {h}" role="img" aria-label="處理流程圖">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         '<path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>']
    for i, (t, side) in enumerate(main):
        y = i * (H + gap) + 2
        hl = " hl" if i in (4, 5) else ""
        o.append(f'<rect class="box{hl}" x="{BX}" y="{y}" width="{BW}" height="{H}" rx="7"/>'
                 f'<text class="bt" x="{BX + BW / 2}" y="{y + H / 2 + 5}" text-anchor="middle">{esc(t)}</text>')
        if i < len(main) - 1:
            o.append(f'<line class="arrow" x1="{BX + BW / 2}" y1="{y + H}" x2="{BX + BW / 2}" y2="{y + H + gap - 2}" marker-end="url(#ah)"/>')
        if side:
            dashed = " dash" if i in (5, 6) else ""
            o.append(f'<line class="arrow{dashed}" x1="{BX + BW}" y1="{y + H / 2}" x2="{SX - 2}" y2="{y + H / 2}" marker-end="url(#ah)"/>'
                     f'<rect class="sbox{dashed}" x="{SX}" y="{y + 2}" width="{SW}" height="{H - 4}" rx="7"/>'
                     f'<text class="st" x="{SX + SW / 2}" y="{y + H / 2 + 4.5}" text-anchor="middle">{esc(side)}</text>')
    # 黃金標準回饋到規則
    y5, y6 = 5 * (H + gap) + 2, 6 * (H + gap) + 2
    xs, xt, ym = SX + 40, BX + BW - 60, y5 + H + gap / 2
    o.append(f'<path class="arrow dash" d="M{xs},{y6 + 2} L{xs},{ym} L{xt},{ym} L{xt},{y5 + H + 2}" fill="none" marker-end="url(#ah)"/>'
             f'<text class="note" x="{xs + 8}" y="{ym + 11}">校正規則門檻（Wilson 下界）</text>')
    o.append("</svg>")
    return "\n".join(o)


# ───────────────────────── 附錄：實驗參數（提示詞直接讀 ws 的 prompts.yaml 副本，不手抄）
import yaml  # noqa: E402

PROMPTS = yaml.safe_load((SITE.parent / "ref/prompts_ws_copy_2026-11-16.yaml").read_text(encoding="utf-8"))["versions"]
META = json.loads((RUNS / "full/full_nv_20261116_1303/run_meta.json").read_text(encoding="utf-8"))


def kv_table(rows, head=("項目", "設定")):
    return ("<div class=tablewrap><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")


def pre(s):
    return f"<pre><code>{esc(s.strip())}</code></pre>"


cfg, mods, env, R = META["config"], META["config"]["models"], META["env"], META["config"]["rules"]
ZH_CLAP = {"instrument": "樂器", "singing": "歌唱", "speech": "說話", "noise": "雜音", "mixed": "混合", "silence": "靜音"}
clap_rows = [(ZH_CLAP.get(c, c), f"<code>{esc(p)}</code>") for c, p in PROMPTS["v3"]["clap"]]
v4 = PROMPTS["v4"]["alm"]
gloss = "；".join(f"{k} → {v.strip()}" for k, v in {**v4["medium_gloss"], **v4["genre_gloss"]}.items())
syn_rows = [(f, f"<code>{esc(k)}</code>", esc("、".join(str(w) for w in ws)))
            for f, t in v4["synonyms"].items() for k, ws in t.items()]

appendix = f"""
<h2 id="params">附錄：實驗參數</h2>
<p>以下數值與文字取自實際執行的設定檔與提示詞檔（工作站 <code>configs/full_instrumental.yaml</code>、<code>prompts.yaml</code>、
<code>run_meta.json</code>），不是事後整理。</p>

<h3>A. 全量執行（下載清單的來源）</h3>
{kv_table([
    ("執行編號", f"<code>{META['run_id']}</code>"),
    ("切窗", f"窗長 {cfg['segmentation']['window_s']:g} 秒、步距 {cfg['segmentation']['hop_s']:g} 秒；只對伴奏軌（no_vocals）跑模型，人聲軌只計算音量"),
    ("AST", f"<code>{mods['ast']['id']}</code>，revision <code>{mods['ast']['revision'][:8]}</code>；輸入 16 kHz 單聲道；batch {mods['ast']['batch']}；527 類各取 sigmoid"),
    ("CLAP", f"<code>{mods['clap']['id']}</code>，revision <code>{mods['clap']['revision'][:8]}</code>；輸入 48 kHz；batch {mods['clap']['batch']}；音訊與 14 句提示詞的 cosine 相似度，類別取第一名提示詞所屬類別；匯出的機率以 softmax（logit scale 100）計算"),
    ("前 k 名", f"AST 與 CLAP 各 {cfg['topk']} 名"),
    ("CLAP 提示詞版本", f"v{META['prompt_version'].lstrip('v')}（見 C）"),
    ("隨機種子", str(cfg["seed"])),
    ("環境", f"Python {env['python']}、torch {env['torch']}、transformers {env['transformers']}、{env['gpu']}"),
    ("耗時", f"1,801 首約 {sum(META['timing_s'].values()) / 60:.0f} 分鐘（讀檔與重採樣 {META['timing_s']['load_resample'] / 60:.0f}、AST {META['timing_s']['ast'] / 60:.0f}、CLAP {META['timing_s']['clap'] / 60:.0f} 分鐘）；GPU 峰值 {META['gpu_peak_allocated_gib']} GiB"),
])}

<h3>B. 判定規則</h3>
<p>AST 分數分成三組：<b>Music</b>（AudioSet 索引 137）、<b>樂器</b>（Musical instrument 子樹，索引 138～213，排除 Bicycle bell、Tuning fork；含打擊樂與鑼，所以只有鑼鼓也算樂器）、<b>人聲</b>（Speech、Male／Female／Child speech、Conversation、Narration、Singing、Choir、Chant、Male／Female／Child singing、Humming、Vocal music 等），每組取最大值。</p>
{kv_table([
    ("靜音", f"RMS &lt; {R['silence_db']:g} dB"),
    ("純樂器（伴奏軌）", f"max(Music, 樂器) ≥ {R['music_min']:g}，且 人聲 &lt; {R['low']:g}，且 CLAP 第一名屬「樂器」"),
    ("純雜音", f"Music、樂器、人聲皆 &lt; {R['low']:g}，且 CLAP 第一名屬「雜音」"),
    ("純人聲（人聲軌，全量未跑）", f"max(人聲, Music) ≥ {R['voice_min']:g}，且 樂器 &lt; {R['low']:g}，且 CLAP 第一名屬歌唱或說話"),
    ("人聲殘留旗標（伴奏軌）", f"人聲 ≥ {R['leak']:g}，或 CLAP 第一名屬歌唱或說話 → <code>residue = vocal_leak</code>"),
    ("混合", "以上皆不成立"),
    ("邊界案例（挑黃金標準用）", f"AST 分數與門檻比值在 {cfg['borderline']['ast_ratio']:g} 倍內、CLAP 第一與第二類差 &lt; {cfg['borderline']['clap_margin']:g}、RMS 在 {cfg['borderline']['rms_band_db'][0]:g}～{cfg['borderline']['rms_band_db'][1]:g} dB"),
], head=("類別", "條件"))}

<h3>C. CLAP 提示詞（v3）</h3>
<p>改版紀錄：v2 用了「old opera recording」這類年代字眼，會把雜音段吸成人聲，v3 刪除；載體一律寫 vinyl；樂器以聲音內容描述。</p>
{kv_table(clap_rows, head=("類別", "提示詞"))}

<h3>D. 音訊語言模型（實驗一）</h3>
{kv_table([
    ("樣本", "305 段：第 2 輪黃金標準候選窗的兩條軌（217 段），加上其餘 pilot 窗依「軌 × 規則類別 × 載體」分層、每格 4 段（88 段）；種子 20261115"),
    ("AF-Next", "<code>nvidia/audio-flamingo-next-hf</code>，revision <code>5634886e</code>；語言模型 8-bit 量化（音訊編碼器與投影層不量化），bf16；輸入 16 kHz；貪婪解碼；描述與抽取各最多 120 tokens"),
    ("Audex-2B", "<code>nvidia/Nemotron-Labs-Audex-2B</code>，revision <code>77b7e1a4</code>；bf16；16 kHz、補零到 30 秒；貪婪解碼；描述 160、抽取 200 tokens；NVIDIA OneWay Noncommercial 授權（研究用途）"),
    ("流程", "第一段聽音訊、只描述；第二段只給描述文字、不給音訊、不給 metadata，輸出 JSON"),
    ("對照組", "描述時帶或不帶 metadata（提示詞 <code>describe</code> 與 <code>describe_meta</code>）"),
])}
<p><b>描述（不帶 metadata）</b></p>{pre(v4['describe'])}
<p><b>描述（帶 metadata）</b>：<code>{{context}}</code> 由模板 <code>{esc(v4['context'])}</code> 填入。</p>{pre(v4['describe_meta'])}
<p class="small">對照表：{esc(gloss)}</p>
<p><b>抽取</b>：<code>{{stem}}</code> 為 vocals 或 accompaniment；<code>{{other}}</code> 在人聲軌為 instruments，在伴奏軌為 voice (singing or speech)。以下是實際送出的文字（設定檔裡的 <code>{{{{</code> 是格式化跳脫，送出時為單一大括號）。</p>{pre(v4['extract'].replace('{{', '{').replace('}}', '}'))}
<p><b>同義詞對照</b>：模型輸出先轉小寫，再依下表歸為標準值；對不上的值記為解析失敗。</p>
{kv_table(syn_rows, head=("欄位", "標準值", "接受的寫法"))}

<h3>E. 本機文字模型抽取（實驗二）</h3>
{kv_table([
    ("輸入", "AF-Next 的 610 筆描述（305 段 × 帶或不帶 metadata），不重聽音訊"),
    ("模型", "、".join(f"<code>{esc(n)}</code>" for n in llm) + "（在筆電以 LM Studio 執行；不使用中國開發的模型）"),
    ("提示詞", "與 D 的抽取提示詞逐字相同"),
    ("解碼", "temperature 0；gpt-oss 加 reasoning_effort = low"),
    ("輸出限制", "JSON schema：cls 與 noise 為列舉值，vocal_type 可為 null，leak 為布林，evidence 最多 120 字元；五欄皆必填"),
    ("evidence 逐字檢查", "去掉頭尾句點、轉小寫後，必須是描述的子字串"),
])}
"""

# ───────────────────────── 頁面
index =(SITE / "index.html").read_text(encoding="utf-8")
style = re.search(r"<style>(.*?)</style>", index, re.S).group(1)
script = re.search(r'(<div id="tip".*?</script>)', index, re.S).group(1)
af_a, af_b, axs = (stats(g) for g in sources.values())
n_llm = len(llm)

body = f"""
<div class="card"><b>這頁寫給沒有參與討論的人。</b>首頁只有結果；這裡說明我們想解決什麼問題、流程每一步怎麼做、為什麼這樣決定，以及到目前為止做過的實驗與結論。
所有類別都還沒有經過人工標註驗證，數字表示模型之間的一致程度，不是正確率。</div>

<nav class="toc"><b>目錄</b><ol>
<li><a href="#goal">目的：要找出什麼樣的片段</a></li>
<li><a href="#data">資料</a></li>
<li><a href="#pipeline">流程</a></li>
<li><a href="#decisions">關鍵決定與理由</a></li>
<li><a href="#gold">黃金標準：怎麼驗證</a></li>
<li><a href="#alm">實驗一：讓音訊語言模型聽，作為審查</a></li>
<li><a href="#extract">實驗二：換一個文字模型來抽取</a></li>
<li><a href="#window">實驗三：窗長 4、8、10 秒</a></li>
<li><a href="#use">使用下載清單的建議</a></li>
<li><a href="#next">目前結論與下一步</a></li>
<li><a href="#params">附錄：實驗參數（模型、門檻、提示詞全文）</a></li>
</ol></nav>

<h2 id="goal">1. 目的：要找出什麼樣的片段</h2>
<p>我們手上有一批日治時期到戰後的臺灣唱片數位檔，已經用 demucs（htdemucs）分離成人聲軌與伴奏軌。目標是從分離後的軌裡，<b>找出可以直接拿來當訓練素材的片段</b>：</p>
<ul>
<li><b>純樂器</b>：只有樂器，沒有人聲殘留。可以當作分離或生成模型的器樂素材。</li>
<li><b>純雜音</b>：只有唱片表面雜音（嘶聲、爆音、低頻隆隆聲），沒有音樂。可以當作資料增強時加入的真實雜音。</li>
<li>人聲軌也有對應的類別（純人聲），但目前全量只跑伴奏軌。</li>
</ul>
<p><b>原則是精確率優先</b>：寧可少收，也不要把帶人聲殘留的片段當成純樂器。訓練素材裡混進錯誤樣本，傷害比少一點素材大。所以只要模型之間有疑義，段落一律歸入「混合」，不進下載清單。</p>

<h2 id="data">2. 資料</h2>
<ul>
<li><b>音檔</b>：1,801 首，伴奏軌合計約 124.6 小時；WAV、44.1 kHz。路徑格式 <code>&lt;唱片公司資料夾&gt;/htdemucs/&lt;曲名&gt;/no_vocals.wav</code>。</li>
<li><b>載體</b>：戰前 78 轉為主，另有日文 78 轉、戰後 78 轉、黑膠與錄音帶。不同載體的雜音特性不同，所有輸出都帶 <code>medium</code> 欄。</li>
<li><b>曲種與年代</b>：約一半的曲目可以對應到唱片目錄（林太崴整理的 metadata），得知曲種（流行歌、歌仔戲、講談、南管、北管等）與錄音年代。這些資訊<b>只用在分層抽樣與統計</b>，不給模型看，理由見下方實驗一。</li>
</ul>

<h2 id="pipeline">3. 流程</h2>
{flow_svg()}
<p class="small">實線是全量管線；虛線是審查與驗證（不參與全量判定）。深色框的模型與規則參數見<a href="#params">附錄</a>。</p>
<ol>
<li><b>切窗</b>：每首切成 8 秒窗、步距 4 秒（相鄰兩窗重疊一半）。</li>
<li><b>靜音</b>：窗的音量（RMS）低於 -60 dB 直接判為靜音。</li>
<li><b>兩個模型打分</b>：
  <ul><li><b>AST</b>（MIT/ast-finetuned-audioset-10-10-0.4593）：在 AudioSet 的 527 類上給分，我們看 Music、各樂器類、人聲類的分數。</li>
  <li><b>CLAP</b>（laion/clap-htsat-unfused）：零樣本分類，拿音訊和一組我們寫的英文描述句比相似度（例如「a bowed two-string Chinese fiddle playing a melody」「vinyl crackle and surface noise, with no music」）。</li></ul>
  兩個模型各保留前 10 名，存在 <code>windows_top10.parquet</code>，讓使用者可以自訂門檻。</li>
<li><b>規則判類</b>：兩個模型都同意才給純類別。純樂器：音樂或樂器分數 ≥ 0.2、人聲分數 &lt; 0.05、CLAP 第一名是樂器句。純雜音：音樂、樂器、人聲分數都 &lt; 0.05，CLAP 第一名是雜音句。其餘一律「混合」。</li>
<li><b>合併</b>：同一首裡相鄰的同類窗合併成連續區段，就是 <code>segments_pure</code> 的每一列。</li>
<li><b>驗證</b>（進行中）：人工試聽一批分層抽樣的窗，建立黃金標準，再回頭校正門檻（見第 5 節）。</li>
</ol>
<p>流程的入口是單一腳本與一份設定檔，環境、套件版本與模型版本都已固定，任何一次執行都可以重跑出相同結果。完整參數與提示詞全文見<a href="#params">附錄</a>。</p>

<h2 id="decisions">4. 關鍵決定與理由</h2>
<div class=tablewrap><table><thead><tr><th>決定</th><th>理由</th></tr></thead><tbody>
<tr><td>主判斷用 AST＋CLAP 加規則，不用大型語言模型</td><td>兩者快（全量 1,801 首約 1 小時），而且輸出連續分數，可以依人工標註設定門檻、控制精確率。音訊語言模型只輸出文字，沒有信心分數，全量也要跑約 8 天（見實驗一）。</td></tr>
<tr><td>只收兩個模型都同意的段落</td><td>精確率優先。兩個模型的錯誤型態不同，同時出錯的機率較低。</td></tr>
<tr><td>提示詞寫「vinyl」，不寫 shellac 或 78 rpm，也不寫「old opera recording」這類年代字眼</td><td>試跑發現，句子裡帶「老錄音」「歌劇」的提示詞會把雜音段吸過去，判成有人聲。模型對罕見詞的理解也不可靠。</td></tr>
<tr><td>樂器以聲音內容描述（拉弦、彈撥、嗩吶、鑼鼓），不用樂種名稱</td><td>模型不認得「北管」「南管」，但聽得出拉弦樂器或打擊樂。單獨的鑼鼓也算樂器。</td></tr>
<tr><td>曲種、年代等 metadata 不給模型</td><td>實驗一證實：給了 metadata，模型會依文字而不是依聲音作答，把純雜音說成有人在唱。</td></tr>
<tr><td>8 秒窗</td><td>試跑時 10 秒窗讓 AST 在靜音段的 Music 分數升高，4 秒窗讓 CLAP 不穩；8 秒兩者都穩定。尚待黃金標準確認，所以 4 與 10 秒也跑全量對照（實驗三）。</td></tr>
<tr><td>保留 track 與曲目分組</td><td>同一首曲目的不同窗、同一齣戲的不同面高度相關。切訓練與測試集時要依曲目分組，否則會洩漏。</td></tr>
</tbody></table></div>

<h2 id="gold">5. 黃金標準：怎麼驗證</h2>
<p>到目前為止所有比較都是「模型對模型」，沒有人知道誰對。黃金標準是人工試聽後給的標籤，用途有兩個：校正規則門檻，讓每一類的精確率 95% 信賴下界（Wilson 下界）達到 0.95；以及評比不同方法、不同窗長誰比較準。</p>
<h3>Wilson 下界是什麼</h3>
<p>從某一類（例如純樂器）抽 n 段來聽，k 段確實正確，樣本精確率是 k/n。樣本有限，真正的精確率可能更高或更低；Wilson 信賴區間（Wilson score interval）給出 95% 的範圍，我們只看<b>下界</b>：有 95% 的把握，真正的精確率不低於它。</p>
<p>不用課本常見的 p̂ ± 1.96√(p̂(1−p̂)/n)，是因為那個公式在比例接近 1、樣本少時會失靈：聽 20 段全對，它給出寬度為 0 的區間，等於宣稱精確率確定是 100%；Wilson 下界則是 0.84，誠實反映樣本太少。要讓下界達到 0.95，需要的樣本數：</p>
<div class=tablewrap><table><thead><tr><th class=num>抽聽中判錯</th><th class=num>至少要聽</th></tr></thead><tbody>
<tr><td class=num>0 段</td><td class=num>73 段</td></tr><tr><td class=num>1 段</td><td class=num>110 段</td></tr>
<tr><td class=num>2 段</td><td class=num>142 段</td></tr><tr><td class=num>3 段</td><td class=num>173 段</td></tr></tbody></table></div>
<p>若某一類聽 100 段錯 5 段，下界只有 0.89，就把該類的門檻收緊、少收一些段落，直到下界過 0.95。這就是「依黃金標準校正門檻」的具體做法。保留：Wilson 假設各段獨立，但重疊窗與同一首內的段落高度相關，實際不確定性更大；所以抽黃金標準時每首最多取 2 窗、分散到不同曲目。</p>
<h3>兩輪抽樣</h3>
<ul>
<li><b>第 1 輪</b>：120 個 8 秒窗，依曲種 × 年代分層隨機抽樣，約兩成取自曲前曲後的空轉段（雜音最集中的地方）。</li>
<li><b>第 2 輪</b>：120 個窗，專挑邊界案例：AST 與 CLAP 意見不同、分數接近門檻、音量接近靜音線、音訊語言模型與規則不一致等。標註時看不到模型的判斷，避免先入為主。</li>
<li>兩條軌各標一次，類別與上方相同，另標人聲型態（唱、說）、分離殘留、雜音程度。試聽工具在實驗室內部網路，標註尚未完成。</li>
</ul>

<h2 id="alm">6. 實驗一：讓音訊語言模型聽，作為審查</h2>
<p><b>想法</b>：AST 與 CLAP 只給分數，不會「解釋」。音訊語言模型（audio-language model，ALM）可以聽一段音訊、用文字描述聽到什麼，適合拿來審查規則判定的邊界案例。做法參考線上教學《LLMs × Music》的〈LLM-based Music Annotation〉（<a href="https://bkwmt.github.io/research-notes/notes/music-llm-annotation-demo.html">中文導讀</a>）：</p>
<ol><li><b>描述</b>：模型聽音訊，用兩三句話描述樂器、人聲、雜音、靜音。貪婪解碼，重跑結果相同。</li>
<li><b>抽取</b>：只把描述文字交給模型，填成固定欄位的 JSON：類別、人聲型態、殘留、雜音程度，以及一段「逐字抄自描述」的證據片語。</li></ol>
<p><b>樣本</b>：305 段（第 2 輪黃金標準候選窗的兩條軌，加上分層抽樣）。<b>對照組</b>：描述時帶或不帶 metadata（載體、曲種、年代）。另外換一個較小的模型 Audex-2B（NVIDIA，2B 參數，非商業授權）做第二意見。</p>
{legend}
{dist_svg}
{src_table}
<p class="small">「欄位全部有效」要求五個欄位都對得上；AF-Next 的失敗多半只是殘留欄輸出 null，類別仍然有效，所以一致率以「類別有值」的段落計算。規則判為純雜音的段落共 {n_noise} 段。</p>

<h3>發現一：給 metadata 會讓模型憑文字作答</h3>
<p>規則判為純雜音的 {n_noise} 段，不帶 metadata 時 AF-Next 有 {af_a['noise_hit']} 段判為純雜音；帶了 metadata 只剩 {af_b['noise_hit']} 段。同一段只有低頻隆隆聲的伴奏軌：</p>
<blockquote><b>不帶</b>：{esc(desc_of(af, EX_SEG, 'a'))}</blockquote>
<blockquote><b>帶</b>（提示詞前加上「{esc(ex_meta)}」）：{esc(desc_of(af, EX_SEG, 'b'))}</blockquote>
<p>模型看到「Taiwanese opera」就描述出男高音和古琴。結論：<b>描述階段一律不給 metadata</b>。metadata 最多只在確認有人聲之後，用來判斷是唱還是說（講談類在帶 metadata 時全部判為說話）。</p>

<h3>發現二：小模型的描述撐不起抽取</h3>
<p>Audex-2B 快約 6 倍（每段約 {axs['sec']:.1f} 秒），但描述很短，常編出不存在的聲音。規則判為純雜音的段落，它的描述例如：</p>
<ul>{"".join(f"<li>{esc(d)}</li>" for d in ax_pick)}</ul>
<p>它也傾向判「混合」（伴奏軌過半），和規則的一致率只有 {pct(axs['rule_agree'])}，純雜音段只抓到 {axs['noise_hit']}／{n_noise}。結論：<b>不採用 Audex-2B</b>。若要第二個音訊模型，應找音樂專用的大模型。</p>

<h3>發現三：音訊語言模型不適合當主判斷</h3>
<p>AF-Next 每段約 {af_a['sec']:.0f} 秒，全量 11 萬段要約 8 天；輸出沒有信心分數，無法設門檻；帶與不帶 metadata 的類別一致率只有六成。所以它的角色是<b>審查邊界案例</b>，不是標註全量。</p>

<h2 id="extract">7. 實驗二：換一個文字模型來抽取</h2>
<p><b>想法</b>：第二段抽取只看文字，可以換成更強的文字模型。我們拿 AF-Next 已經寫好的 610 筆描述（305 段 × 帶或不帶 metadata），交給筆電上的 {"、".join(llm)} 重新抽取。提示詞與原本逐字相同，只多加一條 JSON schema，強制輸出合法 JSON。</p>
{ex_table}
<ul>
<li><b>格式問題消失</b>：schema 強制下 JSON 全部有效；證據片語逐字出自描述的比例大幅提高，「抽取結果必須能在描述裡找到根據」這道檢查才真的能用。</li>
<li><b>類別大致不變</b>：類別主要由描述決定，換抽取模型改變不大。主要差異是 AF-Next 判純人聲、本機模型改判混合：{"；".join(shift)}。描述裡提到隱約伴奏時，本機模型照定義歸為混合，比較嚴格。</li>
<li><b>聽錯救不回來</b>：帶 metadata 時雜音已經被描述成音樂，換什麼抽取模型都一樣判錯。描述階段不給 metadata，比換抽取模型重要得多。</li>
{pair_html}
</ul>

<h2 id="window">8. 實驗三：窗長 4、8、10 秒</h2>
<p>窗太短，模型資訊不足、分數不穩；窗太長，短暫的人聲殘留會被平均掉，或把靜音和音樂混在一起。下載清單的主版本（<code>segments_pure</code>、<code>windows_top10</code>）是 8 秒窗。</p>
{win_html}

<h2 id="use">9. 使用下載清單的建議</h2>
<ul>
<li><b>先抽聽</b>：類別未經人工驗證。建議先從每個載體各抽 20 到 30 段聽過，再決定要不要整批使用。</li>
<li><b>想更保守就收緊門檻</b>：<code>segments_pure</code> 帶有區段內最差的分數，例如只取 <code>ast_voice_max &lt; 0.02</code> 的純樂器段，可以進一步降低人聲殘留的風險；<code>windows_top10</code> 可以完全自訂規則。</li>
<li><b>要更保守可取窗長交集</b>：<code>windows_class_w4／w8／w10.parquet</code> 是三種窗長的逐窗類別（欄位 stem_path、start、end、class、residue）。只取三者都判同一純類別的秒數，時數會減少，但較不受窗長影響。</li>
<li><b>依曲目分組切分</b>：同一首（同一個 <code>stem_path</code>）的區段不要同時出現在訓練與測試集。</li>
<li><b>純雜音的來源偏向曲前曲後</b>：空轉段的雜音和曲中音樂底下的雜音特性可能不同，需要時可依 <code>start</code> 與曲長區分。</li>
<li><b>分離殘留</b>：伴奏軌的錯誤主要是人聲殘留；demucs 對某些樂器（例如洞簫）會誤放進人聲軌，所以伴奏軌的純樂器段可能缺了這類樂器。</li>
</ul>

<h2 id="next">10. 目前結論與下一步</h2>
<ol>
<li>主判斷維持 AST＋CLAP＋規則；音訊語言模型只審查邊界案例，描述時不給 metadata。</li>
<li>第二段抽取改用本機文字模型加 JSON schema，取代 AF-Next 自己抽取。</li>
<li>完成黃金標準兩輪標註後：校正門檻，並在同一批段落上比較規則、AF-Next、本機抽取與三種窗長的精確率，再決定下載清單的最終版本。</li>
<li>人聲軌目前只算音量，之後視需要補跑模型。</li>
</ol>
{appendix}
"""

page = f"""<!doctype html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>工作流程與實驗</title>
<meta name="description" content="戰前 78 轉分軌標註的目的、流程、決策理由，以及音訊語言模型、抽取模型與窗長的實驗紀錄。">
<style>{style}
.viz-root{{--c-vocal:var(--pink)}}
blockquote{{margin:.6em 0;padding:.5em 1em;border-left:3px solid var(--rule);background:var(--card);font-size:.92rem}}
nav.toc{{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px 18px;margin:18px 0}}
nav.toc ol{{margin:.3em 0 0;padding-left:1.4em}}
.small{{font-size:.86rem;color:var(--grey)}}
pre{{white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;font-size:.84rem}}
.flow .box{{fill:var(--card);stroke:var(--line);stroke-width:1.2}}.flow .box.hl{{fill:var(--paper);stroke-width:2}}
.flow .sbox{{fill:var(--bg);stroke:var(--grey);stroke-width:1}}.flow .dash{{stroke-dasharray:4 3}}
.flow .bt{{fill:var(--ink);font-size:13.5px}}.flow .st{{fill:var(--ink);font-size:12px}}.flow .note{{fill:var(--accent);font-size:11.5px}}
.flow .arrow{{stroke:var(--grey);stroke-width:1.3}}.flow .ahead{{fill:var(--grey)}}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
<p class="crumb"><a href="index.html">戰前 78 轉分軌標註</a> › 工作流程與實驗</p>
<h1>工作流程、決策與實驗紀錄</h1>
<p class="sub">我們想從分離後的唱片軌裡找出什麼、怎麼找、為什麼這樣找，以及到目前為止試過什麼。</p>
<p class="meta">更新 2026-11-16</p>
</div></header>
<main class="wrap viz-root">
{body}
</main>{script}</body></html>
"""
(SITE / "workflow.html").write_text(page, encoding="utf-8")
print("workflow.html", len(page) // 1024, "KB；抽取模型：", list(llm))
