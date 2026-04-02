import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import io

from model.train import train, build_feature_matrix, EMOJI_SENTIMENT, extract_emojis
from model.predict import load_model, predict, check_wellbeing, MOOD_META
from model.sarcasm import load_sarcasm_model, detect_sarcasm, train_sarcasm_model

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emoji Mood Analyzer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.mood-card { padding:1.2rem 1.5rem; border-radius:12px; margin-bottom:1rem; }
.sarc-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:0.82rem; font-weight:600; margin-left:10px;
}
.signal-pill {
    display:inline-block; background:#fff3cd; color:#856404;
    border-radius:12px; padding:2px 10px; font-size:0.78rem; margin:2px;
}
</style>
""", unsafe_allow_html=True)

# ─── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading mood model...")
def get_mood_model():
    path = "model/artifacts/model.pkl"
    if not os.path.exists(path):
        data_path = "data/real_data_sample.csv" if os.path.exists("data/real_data_sample.csv") else "data/sample_data.csv"
        train(data_path)
    return load_model(path)

@st.cache_resource(show_spinner="Loading sarcasm model...")
def get_sarcasm_model():
    path = "model/artifacts/sarcasm_model.pkl"
    if not os.path.exists(path):
        return train_sarcasm_model()
    return load_sarcasm_model()

artifacts     = get_mood_model()
sarcasm_model = get_sarcasm_model()

# ─── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Emoji Mood Analyzer")
    st.caption("GoEmotions · Ekman 6 · Sarcasm detection · Batch analysis")
    st.divider()

    data_file = "data/real_data_sample.csv" if os.path.exists("data/real_data_sample.csv") else "data/sample_data.csv"
    df_info   = pd.read_csv(data_file)
    st.subheader("Dataset")
    st.markdown(f"""
- **Source:** {"GoEmotions (Google)" if "real" in data_file else "Sample data"}
- **Size:** {len(df_info):,} samples
- **Classes:** 6 Ekman emotions
- **Best model:** {artifacts.get('model_name', '—')}
    """)
    st.divider()

    st.subheader("Quick samples")
    samples = {
        "😊 Joy":      "Got promoted today, feeling amazing!! 😊🎉✨",
        "😢 Sadness":  "Missing my family so much lately 😢💔",
        "😡 Anger":    "They lied straight to my face again 😡🤬",
        "😨 Fear":     "Waiting for biopsy results 😰😨",
        "😲 Surprise": "Wait WHAT they offered double salary 😲🤯",
        "🤢 Disgust":  "That behaviour is absolutely nauseating 🤮😒",
        "🙄 Sarcasm":  "Oh great, ANOTHER Monday 😒 how wonderful",
    }
    for label, sample in samples.items():
        if st.button(label, use_container_width=True, key=f"s_{label}"):
            st.session_state["prefill"] = sample

    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
st.title("Emoji-Based Mood Analyzer")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analyze", "Mood Trends", "Emoji Heatmap", "Batch Analyzer", "Model Insights"
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — Single text analyzer
# ═══════════════════════════════════════════════════════════
with tab1:
    default_text = st.session_state.pop("prefill", "")
    user_input   = st.text_area(
        "Enter text with emojis", value=default_text, height=100,
        placeholder="e.g.  Finally finished my project 🙌😄✨",
    )
    analyze_clicked = st.button("Analyze mood", type="primary")

    if analyze_clicked and user_input.strip():
        result = predict(user_input, artifacts)
        sarc   = detect_sarcasm(user_input, sarcasm_model)
        result["sarcasm"] = sarc
        st.session_state.history.append(result)

        mood = result["predicted_mood"]
        conf = result["confidence"]
        meta = MOOD_META.get(mood, {"color":"#888","bg":"#eee","icon":"🤔"})

        if sarc["is_sarcastic"]:
            sarc_html = f'<span class="sarc-badge" style="background:#fff3cd;color:#856404">🙄 Sarcasm detected ({sarc["confidence"]}%)</span>'
        else:
            sarc_html = f'<span class="sarc-badge" style="background:#d1e7dd;color:#0a3622">✓ Sincere ({100-sarc["confidence"]:.0f}% confident)</span>'

        st.markdown(f"""
        <div class="mood-card" style="background:{meta['bg']};border-left:5px solid {meta['color']}">
            <span style="font-size:2.4rem">{meta['icon']}</span>
            <span style="font-size:1.5rem;font-weight:600;color:{meta['color']};margin-left:8px">
                {mood.capitalize()}
            </span>
            <span style="font-size:1rem;color:#555;margin-left:10px">{conf}% confidence</span>
            {sarc_html}
        </div>
        """, unsafe_allow_html=True)

        if sarc["signals"]:
            pills = " ".join(f'<span class="signal-pill">{s}</span>' for s in sarc["signals"])
            st.markdown(f"**Sarcasm signals:** {pills}", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("Confidence per mood")
            probs  = result["all_probabilities"]
            msort  = sorted(probs, key=probs.get, reverse=True)
            vals   = [probs[m] for m in msort]
            clrs   = [MOOD_META.get(m, {}).get("color","#aaa") for m in msort]
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars   = ax.barh(msort, vals, color=clrs, height=0.55)
            ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
            ax.set_xlim(0, 120)
            ax.invert_yaxis()
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_r:
            breakdown = result["emoji_breakdown"]
            if breakdown:
                st.subheader("Emoji signals")
                for item in breakdown[:6]:
                    mc = MOOD_META.get(item["mood_signal"], {}).get("color","#888")
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:6px 10px;
                                margin-bottom:6px;background:#f8f9fa;border-radius:8px">
                        <span style="font-size:1.5rem">{item['emoji']}</span>
                        <div>
                            <div style="font-size:0.78rem;color:#555">{item['name'][:28]}</div>
                            <div style="font-size:0.82rem;font-weight:600;color:{mc}">
                                {item['mood_signal']} · weight {item['weight']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No emoji signals found in lexicon.")

        ef = result["emoji_features"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Emojis found",    ef["emoji_count"])
        c2.metric("Unique",          ef["unique_emojis"])
        c3.metric("Emoji density",   ef["emoji_density"])
        c4.metric("Sentiment score", ef["sentiment_score"])

        mood_seq = [r["predicted_mood"] for r in st.session_state.history]
        wb = check_wellbeing(mood_seq)
        if wb["alert"]:
            st.warning(f"Well-being notice — {wb['message']} (streak: {wb['streak']})")

    elif analyze_clicked:
        st.warning("Please enter some text.")


# ═══════════════════════════════════════════════════════════
# TAB 2 — Mood Trend Over Time
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Mood trend over time")

    if len(st.session_state.history) < 2:
        st.info("Analyze at least 2 texts in the Analyze tab to see trends here.")
    else:
        history  = st.session_state.history
        mood_seq = [r["predicted_mood"] for r in history]
        conf_seq = [r["confidence"] for r in history]
        sarc_seq = [r.get("sarcasm", {}).get("is_sarcastic", False) for r in history]

        st.markdown("**Session timeline**")
        cols = st.columns(min(len(history), 12))
        for i, r in enumerate(history[-12:]):
            m    = r["predicted_mood"]
            meta = MOOD_META.get(m, {"icon":"?","color":"#888","bg":"#eee"})
            flag = "🙄" if r.get("sarcasm", {}).get("is_sarcastic") else ""
            with cols[i]:
                st.markdown(f"""
                <div style="text-align:center;background:{meta['bg']};border-radius:8px;padding:8px 4px">
                    <div style="font-size:1.5rem">{meta['icon']}{flag}</div>
                    <div style="font-size:0.62rem;color:{meta['color']};font-weight:600">{m}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Confidence over session**")
            c_seq = [MOOD_META.get(m,{}).get("color","#aaa") for m in mood_seq]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for i in range(len(conf_seq)-1):
                ax.plot([i,i+1],[conf_seq[i],conf_seq[i+1]], color=c_seq[i], lw=1.8)
            ax.scatter(range(len(conf_seq)), conf_seq, c=c_seq, s=55, zorder=5)
            for i, is_s in enumerate(sarc_seq):
                if is_s:
                    ax.annotate("🙄", (i, conf_seq[i]),
                                textcoords="offset points", xytext=(0,8),
                                ha="center", fontsize=9)
            ax.set_ylim(0, 115)
            ax.set_xlabel("Entry #")
            ax.set_ylabel("Confidence %")
            ax.spines[["top","right"]].set_visible(False)
            patches = [mpatches.Patch(color=MOOD_META[m]["color"], label=m)
                       for m in MOOD_META if m in mood_seq]
            ax.legend(handles=patches, fontsize=8, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_r:
            st.markdown("**Mood distribution (last 10)**")
            window_moods = mood_seq[-10:]
            counts = Counter(window_moods)
            clrs   = [MOOD_META.get(m,{}).get("color","#aaa") for m in counts]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.pie(counts.values(), labels=counts.keys(), colors=clrs,
                   autopct="%1.0f%%", startangle=140,
                   wedgeprops={"width":0.55}, textprops={"fontsize":9})
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        sarc_rate = sum(sarc_seq)/len(sarc_seq)*100
        wb = check_wellbeing(mood_seq)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total entries",   len(history))
        c2.metric("Sarcasm rate",    f"{sarc_rate:.0f}%")
        c3.metric("Negative streak", wb["streak"])
        c4.metric("Dominant mood",   wb["dominant_recent_mood"].capitalize())
        if wb["alert"]:
            st.warning(f"Well-being alert — {wb['message']}")

        st.subheader("Full session log")
        table = []
        for i, r in enumerate(history):
            m = r["predicted_mood"]
            table.append({
                "#": i+1,
                "Text": r["text"][:55]+("…" if len(r["text"])>55 else ""),
                "Mood": f"{MOOD_META.get(m,{}).get('icon','?')} {m}",
                "Confidence": f"{r['confidence']}%",
                "Sarcasm": "Yes 🙄" if r.get("sarcasm",{}).get("is_sarcastic") else "No",
                "Emojis": r["emoji_features"]["emoji_count"],
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — Emoji Frequency Heatmap
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Emoji frequency heatmap")
    st.caption("Which emojis co-occur most with each mood in the training dataset.")

    @st.cache_data
    def build_heatmap_data(data_file):
        df = pd.read_csv(data_file)
        top_emojis = list(EMOJI_SENTIMENT.keys())[:20]
        moods      = ["joy","sadness","anger","fear","surprise","disgust"]
        matrix     = {m: {e: 0 for e in top_emojis} for m in moods}
        for _, row in df.iterrows():
            mood = row["mood"]
            if mood not in moods:
                continue
            for ch in str(row["text"]):
                if ch in top_emojis:
                    matrix[mood][ch] += 1
        hdf = pd.DataFrame(matrix, index=top_emojis).T
        return hdf.loc[:, (hdf != 0).any(axis=0)]

    data_file = "data/real_data_sample.csv" if os.path.exists("data/real_data_sample.csv") else "data/sample_data.csv"
    hmap_df   = build_heatmap_data(data_file)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(hmap_df, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.4, ax=ax, cbar_kws={"label":"Frequency"})
    ax.set_xlabel("Emoji")
    ax.set_ylabel("Mood")
    ax.set_title("Emoji co-occurrence frequency by mood class", fontsize=12)
    ax.set_xticklabels(hmap_df.columns.tolist(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Top emojis per mood")
    mood_cols = st.columns(3)
    for i, mood in enumerate(hmap_df.index):
        meta  = MOOD_META.get(mood, {"color":"#888","bg":"#eee","icon":"?"})
        top5  = hmap_df.loc[mood].sort_values(ascending=False)
        top5  = top5[top5 > 0].head(5)
        items = "".join(
            f'<span style="font-size:1.3rem">{e}</span>'
            f'<span style="font-size:0.78rem;color:#555"> {int(c)}</span>  '
            for e,c in top5.items()
        )
        with mood_cols[i % 3]:
            st.markdown(f"""
            <div style="background:{meta['bg']};border-radius:10px;padding:10px 14px;margin-bottom:10px">
                <div style="font-weight:600;color:{meta['color']};margin-bottom:6px">
                    {meta['icon']} {mood.capitalize()}
                </div>
                <div>{items or "No emoji data"}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — Batch CSV Analyzer
# ═══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Batch CSV analyzer")
    st.caption("Upload any CSV — get mood + sarcasm predictions for every row, then download the results.")

    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    with col_cfg:
        text_col = st.text_input("Text column name", value="text")
        max_rows = st.slider("Max rows to analyze", 10, 500, 100)

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded).head(max_rows)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            raw_df = None

        if raw_df is not None:
            if text_col not in raw_df.columns:
                st.error(f"Column '{text_col}' not found. Available: {list(raw_df.columns)}")
            else:
                st.success(f"Loaded {len(raw_df)} rows — analyzing...")
                progress    = st.progress(0)
                result_rows = []

                for idx, (_, row) in enumerate(raw_df.iterrows()):
                    text = str(row[text_col])
                    r    = predict(text, artifacts)
                    s    = detect_sarcasm(text, sarcasm_model)
                    result_rows.append({
                        "text":            text[:80],
                        "mood":            r["predicted_mood"],
                        "mood_confidence": r["confidence"],
                        "sarcasm":         "Yes" if s["is_sarcastic"] else "No",
                        "sarc_confidence": s["confidence"],
                        "emoji_count":     r["emoji_features"]["emoji_count"],
                        "sentiment_score": r["emoji_features"]["sentiment_score"],
                    })
                    progress.progress((idx+1)/len(raw_df))

                progress.empty()
                result_df = pd.DataFrame(result_rows)

                mood_dist  = Counter(result_df["mood"])
                sarc_count = (result_df["sarcasm"] == "Yes").sum()
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Rows analyzed",   len(result_df))
                c2.metric("Dominant mood",   mood_dist.most_common(1)[0][0].capitalize())
                c3.metric("Sarcasm detected",f"{sarc_count} ({sarc_count/len(result_df)*100:.0f}%)")
                c4.metric("Avg emoji/row",   f"{result_df['emoji_count'].mean():.1f}")

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.markdown("**Mood distribution**")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    mc   = result_df["mood"].value_counts()
                    clrs = [MOOD_META.get(m,{}).get("color","#aaa") for m in mc.index]
                    bars = ax.bar(mc.index, mc.values, color=clrs, width=0.6)
                    ax.bar_label(bars, padding=3, fontsize=9)
                    ax.spines[["top","right"]].set_visible(False)
                    ax.set_ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col_v2:
                    st.markdown("**Confidence distribution**")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    for mood in result_df["mood"].unique():
                        sub = result_df[result_df["mood"]==mood]["mood_confidence"]
                        ax.hist(sub, bins=10, alpha=0.55, label=mood,
                                color=MOOD_META.get(mood,{}).get("color","#aaa"))
                    ax.set_xlabel("Confidence %")
                    ax.legend(fontsize=8)
                    ax.spines[["top","right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.subheader("Results table")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "mood_results.csv", "text/csv",
                    use_container_width=True,
                )
    else:
        st.info("Upload a CSV with a text column. Results + charts appear here.")
        st.code("text\nI love this product 😊\nReally disappointed 😞\nOh great, another delay 😒")


# ═══════════════════════════════════════════════════════════
# TAB 5 — Model Insights
# ═══════════════════════════════════════════════════════════
with tab5:
    st.subheader("Model comparison")
    summary = artifacts.get("results_summary", {})
    if summary:
        rows_perf = [{"Model": k,
                      "CV F1 (macro)": round(v["cv_f1"], 3),
                      "Test F1":       round(v["test_f1"], 3)}
                     for k, v in summary.items()]
        df_perf = pd.DataFrame(rows_perf).sort_values("CV F1 (macro)", ascending=False)

        col_t, col_c = st.columns([2, 3])
        with col_t:
            st.dataframe(df_perf, use_container_width=True, hide_index=True)
        with col_c:
            x  = np.arange(len(df_perf))
            fig, ax = plt.subplots(figsize=(6, 3))
            b1 = ax.bar(x-0.2, df_perf["CV F1 (macro)"], 0.38, label="CV F1",   color="#534AB7", alpha=0.85)
            b2 = ax.bar(x+0.2, df_perf["Test F1"],       0.38, label="Test F1", color="#1D9E75", alpha=0.85)
            ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(df_perf["Model"], fontsize=8, rotation=10)
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.subheader("Confusion matrix")
    cm      = artifacts.get("confusion_matrix")
    classes = artifacts.get("classes", [])
    if cm is not None:
        col_cm, col_info = st.columns([2, 1])
        with col_cm:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes,
                        linewidths=0.5, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(artifacts.get("model_name",""), fontsize=11)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col_info:
            st.markdown(f"""
**Pipeline:**
- TF-IDF (word + bigrams)
- Emoji → Unicode names
- {artifacts.get('model_name','')}

**Dataset:**
- {len(df_info):,} samples
- 6 balanced Ekman classes

**Key insight:**
😊 becomes `smiling_face` — a real TF-IDF token.
            """)

    st.subheader("Emoji sentiment lexicon")
    lex = [{"Emoji":e,"Mood":m,"Weight":w} for e,(m,w) in EMOJI_SENTIMENT.items()]
    st.dataframe(pd.DataFrame(lex), use_container_width=True, height=280, hide_index=True)
