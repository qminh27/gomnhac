"""
Ứng dụng gom cụm nhạc MP3 - dataset FMA (Free Music Archive)
Chạy:  streamlit run app.py
"""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Gom cụm nhạc FMA", page_icon="🎵", layout="wide")

ALGOS = ["K-Means", "Gaussian Mixture", "Agglomerative (Ward)", "DBSCAN", "HDBSCAN"]


# =====================================================================
# 1. DỮ LIỆU + TIỀN XỬ LÝ
# =====================================================================
@st.cache_data(show_spinner="Đang đọc features.csv và tracks.csv ...")
def load_metadata(meta_dir, subset, max_n):
    feats = pd.read_csv(
        os.path.join(meta_dir, "features.csv"), index_col=0, header=[0, 1, 2]
    )
    feats.index = feats.index.astype(int)

    tracks = pd.read_csv(os.path.join(meta_dir, "tracks.csv"), index_col=0, header=[0, 1])
    tracks.index = tracks.index.astype(int)

    allowed = {"small": ["small"], "medium": ["small", "medium"]}[subset]
    tracks = tracks[tracks[("set", "subset")].isin(allowed)]

    info = pd.DataFrame(
        {
            "title": tracks[("track", "title")],
            "artist": tracks[("artist", "name")],
            "genre": tracks[("track", "genre_top")],
        }
    )
    ids = feats.index.intersection(info.index)
    if max_n and 0 < max_n < len(ids):
        ids = pd.Index(np.sort(np.random.RandomState(0).choice(ids, max_n, replace=False)))
    return feats.loc[ids].astype(np.float32), info.loc[ids]


@st.cache_resource(show_spinner="Đang chuẩn hóa + PCA ...")
def build_space(meta_dir, subset, max_n, scaler_name, var):
    feats, info = load_metadata(meta_dir, subset, max_n)
    med = feats.median()
    raw = feats.fillna(med).values
    scaler = StandardScaler() if scaler_name == "StandardScaler" else RobustScaler()
    pipe = Pipeline(
        [("scaler", scaler), ("pca", PCA(n_components=var, svd_solver="full", random_state=0))]
    )
    X = pipe.fit_transform(raw)
    return {"X": X, "pipe": pipe, "median": med, "columns": feats.columns, "info": info}


@st.cache_resource
def get_nn(space_key, n_neighbors=11):
    X = build_space(*space_key)["X"]
    return NearestNeighbors(n_neighbors=n_neighbors).fit(X)


# =====================================================================
# 2. GOM CỤM + ĐÁNH GIÁ
# =====================================================================
def fit_labels(X, algo, k, eps, min_pts, min_cs):
    if algo == "K-Means":
        return KMeans(k, n_init=10, random_state=0).fit_predict(X)
    if algo == "Gaussian Mixture":
        return GaussianMixture(k, covariance_type="diag", random_state=0).fit_predict(X)
    if algo.startswith("Agglomerative"):
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if algo == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
    if algo == "HDBSCAN":
        return HDBSCAN(min_cluster_size=min_cs).fit_predict(X)
    raise ValueError(algo)


@st.cache_data(show_spinner="Đang gom cụm ...")
def run_clustering(space_key, algo, k, eps, min_pts, min_cs):
    X = build_space(*space_key)["X"]
    return fit_labels(X, algo, k, eps, min_pts, min_cs)


def purity(labels, truth):
    ct = pd.crosstab(labels, truth)
    return ct.max(axis=1).sum() / ct.values.sum()


def evaluate(X, labels, genres):
    res = {}
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    res["Số cụm"] = n_clusters
    res["Tỉ lệ nhiễu"] = float(1 - mask.mean())
    if n_clusters >= 2 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        res["Silhouette ↑"] = silhouette_score(
            Xm, lm, sample_size=min(5000, len(lm)), random_state=0
        )
        res["Davies-Bouldin ↓"] = davies_bouldin_score(Xm, lm)
        res["Calinski-Harabasz ↑"] = calinski_harabasz_score(Xm, lm)
    g = genres.values
    gm = mask & pd.notna(g)
    if gm.sum() > 0 and n_clusters >= 1:
        res["ARI ↑"] = adjusted_rand_score(g[gm], labels[gm])
        res["NMI ↑"] = normalized_mutual_info_score(g[gm], labels[gm])
        res["Purity ↑"] = purity(labels[gm], g[gm])
    return res


def cname(l):
    return "Nhiễu" if l == -1 else f"Cụm {l}"


def cnames(labels):
    return np.array([cname(l) for l in labels])


def corder(labels):
    return [cname(l) for l in sorted(set(labels))]


# =====================================================================
# 3. TRỰC QUAN HÓA
# =====================================================================
@st.cache_data(show_spinner="Đang giảm chiều về 2D ...")
def project(space_key, method):
    X = build_space(*space_key)["X"]
    if method == "PCA":
        return X[:, :2]
    if method == "t-SNE":
        return TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X[:, :50])
    if method == "UMAP":
        import umap

        return umap.UMAP(n_components=2, random_state=0).fit_transform(X[:, :50])
    raise ValueError(method)


def available_projections():
    opts = ["PCA", "t-SNE"]
    try:
        import umap  # noqa: F401

        opts.append("UMAP")
    except ImportError:
        pass
    return opts


def scatter2d(coords, color, info, order=None, title=""):
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "color": color,
            "Tên bài": info["title"].values,
            "Nghệ sĩ": info["artist"].values,
            "Thể loại": info["genre"].fillna("N/A").values,
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="color",
        opacity=0.7,
        hover_data=["Tên bài", "Nghệ sĩ", "Thể loại"],
        category_orders={"color": order} if order else None,
        title=title,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text="", height=560, xaxis_title="", yaxis_title="")
    return fig


# =====================================================================
# 4. TRÍCH ĐẶC TRƯNG TỪ MP3 (giống hệt script features.py của FMA -> 518 cột)
# =====================================================================
@st.cache_data(show_spinner="Đang trích đặc trưng từ file nhạc (khoảng 20-60 giây) ...")
def extract_features(data: bytes, suffix: str) -> pd.Series:
    import librosa

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    try:
        dur = librosa.get_duration(path=path)
        offset = max(0.0, dur / 2 - 15)  # lấy 30 giây ở giữa bài (giống FMA clips)
        x, sr = librosa.load(path, sr=None, mono=True, offset=offset, duration=30)
    finally:
        os.remove(path)

    vals = {}

    def add(name, f):
        stat_fns = {
            "mean": lambda a: np.mean(a, axis=1),
            "std": lambda a: np.std(a, axis=1),
            "skew": lambda a: stats.skew(a, axis=1),
            "kurtosis": lambda a: stats.kurtosis(a, axis=1),
            "median": lambda a: np.median(a, axis=1),
            "min": lambda a: np.min(a, axis=1),
            "max": lambda a: np.max(a, axis=1),
        }
        for moment, fn in stat_fns.items():
            for i, v in enumerate(fn(f)):
                vals[(name, moment, f"{i + 1:02d}")] = v

    add("zcr", librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))

    cqt = np.abs(
        librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None)
    )
    add("chroma_cqt", librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7))
    cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    add("chroma_cens", cens)
    add("tonnetz", librosa.feature.tonnetz(chroma=cens))
    del cqt

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    add("chroma_stft", librosa.feature.chroma_stft(S=stft**2, n_chroma=12))
    add("rmse", librosa.feature.rms(S=stft))
    add("spectral_centroid", librosa.feature.spectral_centroid(S=stft))
    add("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft))
    add("spectral_contrast", librosa.feature.spectral_contrast(S=stft, n_bands=6))
    add("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    add("mfcc", librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))

    return pd.Series(vals)


# =====================================================================
# 5. TIỆN ÍCH
# =====================================================================
def audio_path(audio_dir, tid):
    s = f"{int(tid):06d}"
    p = os.path.join(audio_dir, s[:3], s + ".mp3")
    return p if os.path.exists(p) else None


def player(audio_dir, tid):
    p = audio_path(audio_dir, tid)
    if p:
        st.audio(p)
    else:
        st.caption("(Không tìm thấy file mp3 - kiểm tra lại thư mục audio ở sidebar)")


# =====================================================================
# SIDEBAR
# =====================================================================
st.title("🎵 Gom cụm nhạc MP3 với FMA dataset")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    meta_dir = st.text_input("Thư mục fma_metadata", "fma_metadata")
    audio_dir = st.text_input("Thư mục audio (fma_small)", "fma_small")
    subset = st.selectbox("Tập dữ liệu", ["small", "medium"], help="small: 8.000 bài; medium: 25.000 bài")
    max_n = st.number_input("Giới hạn số bài (0 = dùng hết)", 0, 30000, 0, step=500)

    st.subheader("Tiền xử lý")
    scaler_name = st.selectbox("Chuẩn hóa", ["StandardScaler", "RobustScaler"])
    var = st.slider("Phương sai PCA giữ lại", 0.80, 0.99, 0.95, 0.01)

    st.subheader("Thuật toán")
    algo = st.selectbox("Thuật toán gom cụm", ALGOS)
    k = st.slider("Số cụm K", 2, 20, 8) if algo in ALGOS[:3] else 8
    eps = min_pts = min_cs = None
    if algo == "DBSCAN":
        eps = st.slider("eps", 1.0, 50.0, 12.0, 0.5)
        min_pts = st.slider("min_samples", 2, 50, 5)
    else:
        eps, min_pts = 12.0, 5
    if algo == "HDBSCAN":
        min_cs = st.slider("min_cluster_size", 5, 200, 30)
    else:
        min_cs = 30

    run = st.button("🚀 Chạy gom cụm", type="primary", use_container_width=True)

if not os.path.exists(os.path.join(meta_dir, "features.csv")):
    st.error(f"Không tìm thấy `{meta_dir}/features.csv`. Hãy tải fma_metadata.zip và giải nén.")
    st.stop()

if run:
    st.session_state["cfg"] = dict(
        space_key=(meta_dir, subset, int(max_n), scaler_name, var),
        audio_dir=audio_dir,
        algo=algo,
        k=k,
        eps=eps,
        min_pts=min_pts,
        min_cs=min_cs,
    )

cfg = st.session_state.get("cfg")
if cfg is None:
    st.info("👈 Chọn cấu hình ở thanh bên rồi bấm **Chạy gom cụm**.")
    st.stop()

space_key = cfg["space_key"]
space = build_space(*space_key)
X, info, pipe = space["X"], space["info"], space["pipe"]
labels = run_clustering(space_key, cfg["algo"], cfg["k"], cfg["eps"], cfg["min_pts"], cfg["min_cs"])

st.caption(
    f"**{len(X):,}** bài · **{X.shape[1]}** chiều sau PCA (từ 518) · "
    f"thuật toán: **{cfg['algo']}**"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Kết quả", "📈 Chọn tham số", "⚖️ So sánh thuật toán", "🎧 Khám phá cụm", "🔮 Dự đoán MP3 mới"]
)

# ---------------------------------------------------------------------
# TAB 1: KẾT QUẢ
# ---------------------------------------------------------------------
with tab1:
    metrics = evaluate(X, labels, info["genre"])
    cols = st.columns(len(metrics))
    for c, (name, v) in zip(cols, metrics.items()):
        c.metric(name, f"{v:.3f}" if isinstance(v, float) else v)
    with st.expander("Ý nghĩa các độ đo"):
        st.markdown(
            "- **Silhouette** (−1→1), **Calinski-Harabasz**: càng cao càng tốt; **Davies-Bouldin**: càng thấp càng tốt (độ đo nội tại, không cần nhãn).\n"
            "- **ARI, NMI, Purity**: so cụm với thể loại (genre) thật. Giá trị thấp **không có nghĩa là sai**: "
            "gom cụm theo đặc trưng âm thanh không nhất thiết trùng với nhãn thể loại do con người gán."
        )

    c1, c2 = st.columns([3, 1])
    with c1:
        method = st.radio("Giảm chiều 2D bằng", available_projections(), horizontal=True)
        color_by = st.radio("Tô màu theo", ["Cụm", "Thể loại (genre)"], horizontal=True)
        coords = project(space_key, method)
        if color_by == "Cụm":
            fig = scatter2d(coords, cnames(labels), info, corder(labels), f"{method} - màu theo cụm")
        else:
            fig = scatter2d(coords, info["genre"].fillna("N/A").values, info, None, f"{method} - màu theo genre")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sizes = pd.Series(cnames(labels)).value_counts().rename_axis("Cụm").reset_index(name="Số bài")
        st.markdown("**Kích thước cụm**")
        st.dataframe(sizes, hide_index=True, use_container_width=True)

    st.subheader("Phân bố thể loại trong từng cụm")
    ct = pd.crosstab(cnames(labels), info["genre"].fillna("N/A").values, normalize="index")
    ct = ct.loc[[c for c in corder(labels) if c in ct.index]]
    heat = px.imshow(ct, text_auto=".0%", aspect="auto", color_continuous_scale="Blues")
    heat.update_layout(xaxis_title="Thể loại", yaxis_title="", height=420)
    st.plotly_chart(heat, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: CHỌN THAM SỐ
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Elbow + Silhouette cho K-Means")
    kmin, kmax = st.slider("Khoảng K", 2, 30, (2, 15))

    @st.cache_data(show_spinner="Đang chạy K-Means cho nhiều giá trị K ...")
    def sweep_k(space_key, kmin, kmax):
        Xs = build_space(*space_key)["X"]
        rows = []
        for kk in range(kmin, kmax + 1):
            km = KMeans(kk, n_init=5, random_state=0).fit(Xs)
            sil = silhouette_score(Xs, km.labels_, sample_size=min(3000, len(Xs)), random_state=0)
            rows.append((kk, km.inertia_, sil))
        return pd.DataFrame(rows, columns=["K", "Inertia", "Silhouette"])

    if st.button("Tính Elbow / Silhouette"):
        st.session_state["sweep"] = sweep_k(space_key, kmin, kmax)
    if "sweep" in st.session_state:
        sw = st.session_state["sweep"]
        a, b = st.columns(2)
        a.plotly_chart(px.line(sw, x="K", y="Inertia", markers=True, title="Elbow (Inertia)"), use_container_width=True)
        b.plotly_chart(px.line(sw, x="K", y="Silhouette", markers=True, title="Silhouette"), use_container_width=True)
        st.success(f"K có Silhouette cao nhất: **{int(sw.loc[sw.Silhouette.idxmax(), 'K'])}**")

    st.divider()
    st.subheader("k-distance graph (chọn eps cho DBSCAN)")
    kd = st.slider("k (= min_samples)", 2, 50, 5, key="kd")
    if st.button("Vẽ k-distance"):
        d, _ = NearestNeighbors(n_neighbors=kd).fit(X).kneighbors(X)
        fig = px.line(y=np.sort(d[:, -1]), labels={"x": "Điểm (đã sắp xếp)", "y": f"Khoảng cách tới láng giềng thứ {kd}"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Chọn eps tại vị trí 'khuỷu tay' (đường cong bắt đầu dốc lên mạnh).")

# ---------------------------------------------------------------------
# TAB 3: SO SÁNH THUẬT TOÁN
# ---------------------------------------------------------------------
with tab3:
    st.write("Chạy cả 5 thuật toán với tham số hiện tại (K, eps, min_samples, min_cluster_size) rồi so sánh.")
    if st.button("Chạy so sánh"):
        rows = []
        prog = st.progress(0.0)
        for i, a in enumerate(ALGOS):
            if a.startswith("Agglo") and len(X) > 15000:
                rows.append({"Thuật toán": a, "Ghi chú": "Bỏ qua (quá nhiều bài, tốn RAM)"})
            else:
                lab = run_clustering(space_key, a, cfg["k"], cfg["eps"], cfg["min_pts"], cfg["min_cs"])
                rows.append({"Thuật toán": a, **evaluate(X, lab, info["genre"])})
            prog.progress((i + 1) / len(ALGOS))
        st.session_state["compare"] = pd.DataFrame(rows).set_index("Thuật toán")
    if "compare" in st.session_state:
        st.dataframe(st.session_state["compare"].style.format("{:.3f}", na_rep="-"), use_container_width=True)
        st.caption("DBSCAN/HDBSCAN: độ đo tính trên các điểm không phải nhiễu. Hãy chỉnh eps/min_cluster_size nếu ra 0-1 cụm.")

# ---------------------------------------------------------------------
# TAB 4: KHÁM PHÁ CỤM
# ---------------------------------------------------------------------
with tab4:
    opts = [l for l in sorted(set(labels))]
    sel = st.selectbox("Chọn cụm", opts, format_func=cname)
    idx = np.where(labels == sel)[0]
    sub = info.iloc[idx]
    st.write(f"**{len(idx)}** bài trong {cname(sel)}")

    a, b = st.columns([1, 2])
    with a:
        gc = sub["genre"].fillna("N/A").value_counts().rename_axis("Thể loại").reset_index(name="Số bài")
        st.plotly_chart(px.bar(gc, x="Số bài", y="Thể loại", orientation="h", height=320), use_container_width=True)
    with b:
        st.markdown("**🎯 Bài tiêu biểu (gần tâm cụm nhất)**")
        center = X[idx].mean(axis=0)
        near = idx[np.argsort(np.linalg.norm(X[idx] - center, axis=1))[:5]]
        for i in near:
            tid = info.index[i]
            st.write(f"**{info.iloc[i]['title']}** - {info.iloc[i]['artist']} · _{info.iloc[i]['genre']}_")
            player(cfg["audio_dir"], tid)

    st.markdown("**🎲 Nghe ngẫu nhiên**")
    if "seed" not in st.session_state:
        st.session_state["seed"] = 0
    if st.button("Xáo trộn"):
        st.session_state["seed"] += 1
    rnd = np.random.RandomState(st.session_state["seed"]).choice(idx, min(5, len(idx)), replace=False)
    for i in rnd:
        st.write(f"**{info.iloc[i]['title']}** - {info.iloc[i]['artist']} · _{info.iloc[i]['genre']}_")
        player(cfg["audio_dir"], info.index[i])

    with st.expander("Danh sách toàn bộ bài trong cụm"):
        st.dataframe(sub, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 5: DỰ ĐOÁN MP3 MỚI
# ---------------------------------------------------------------------
with tab5:
    st.write(
        "Upload một bài nhạc bất kỳ. Ứng dụng trích đặc trưng (cùng quy trình với FMA), "
        "chiếu vào không gian PCA, rồi gán vào cụm theo **đa số 10 bài gần nhất** (kNN)."
    )
    up = st.file_uploader("Chọn file nhạc", type=["mp3", "wav", "ogg", "flac"])
    if up is not None:
        st.audio(up)
        try:
            feat = extract_features(up.getvalue(), os.path.splitext(up.name)[1] or ".mp3")
        except Exception as e:  # noqa: BLE001
            st.error(f"Không đọc được file: {e}\n\nGợi ý: cài ffmpeg (`pip install librosa soundfile` + ffmpeg) để đọc MP3.")
            st.stop()

        vec = feat.reindex(space["columns"])
        if vec.isna().any():
            vec = vec.fillna(space["median"])
        z = pipe.transform(vec.values.reshape(1, -1).astype(np.float64))

        nn = get_nn(space_key, 11)
        dist, ind = nn.kneighbors(z, n_neighbors=10)
        ind, dist = ind[0], dist[0]
        neigh_labels = labels[ind]
        valid = [l for l in neigh_labels if l != -1] or list(neigh_labels)
        pred = pd.Series(valid).mode().iloc[0]

        st.success(f"### Bài này thuộc **{cname(pred)}**")
        c1, c2 = st.columns([3, 2])
        with c1:
            bg = pd.DataFrame(
                {"x": X[:, 0], "y": X[:, 1], "Cụm": cnames(labels), "Tên bài": info["title"].values, "Thể loại": info["genre"].fillna("N/A").values}
            )
            fig = px.scatter(bg, x="x", y="y", color="Cụm", opacity=0.35, hover_data=["Tên bài", "Thể loại"],
                             category_orders={"Cụm": corder(labels)}, title="Vị trí bài mới (⭐) trong không gian PCA (2 thành phần đầu)")
            fig.add_trace(go.Scatter(x=[z[0, 0]], y=[z[0, 1]], mode="markers",
                                     marker=dict(symbol="star", size=22, color="black", line=dict(width=2, color="white")),
                                     name="Bài mới"))
            fig.update_layout(height=520, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            members = info.iloc[np.where(labels == pred)[0]]
            gc = members["genre"].fillna("N/A").value_counts().rename_axis("Thể loại").reset_index(name="Số bài")
            st.markdown(f"**Thể loại phổ biến trong {cname(pred)}**")
            st.plotly_chart(px.bar(gc, x="Số bài", y="Thể loại", orientation="h", height=320), use_container_width=True)

        st.subheader("🎶 10 bài giống nhất (gợi ý)")
        for r, (i, d) in enumerate(zip(ind, dist), 1):
            row = info.iloc[i]
            st.write(f"**{r}. {row['title']}** - {row['artist']} · _{row['genre']}_ · {cname(labels[i])} · khoảng cách {d:.2f}")
            player(cfg["audio_dir"], info.index[i])
