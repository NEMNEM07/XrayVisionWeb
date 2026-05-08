import { useState, useRef } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    if (f.name.endsWith(".dcm")) {
      setPreview(null);
    } else {
      setPreview(URL.createObjectURL(f));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post(`${API_URL}/predict`, form);
      setResult(res.data);
    } catch (e) {
      setError("분석 중 오류가 발생했습니다. 다시 시도해주세요.");
    } finally {
      setLoading(false);
    }
  };

  const statusColor = result?.status === "ABNORMAL" ? "#ef4444" : "#22c55e";

  return (
    <div style={styles.container}>
      {/* 헤더 */}
      <div style={styles.header}>
        <h1 style={styles.title}>🫁 XrayVision</h1>
        <p style={styles.subtitle}>흉부 X-ray 이상 탐지 AI — ViT-Base/16 기반</p>
      </div>

      {/* 업로드 영역 */}
      <div
        style={styles.dropzone}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".dcm,.png,.jpg,.jpeg"
          style={{ display: "none" }}
          onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
        />
        {file ? (
          <div>
            {preview ? (
              <img src={preview} alt="preview" style={styles.previewImg} />
            ) : (
              <div style={styles.dcmIcon}>📄</div>
            )}
            <p style={styles.fileName}>{file.name}</p>
          </div>
        ) : (
          <div>
            <div style={styles.uploadIcon}>📂</div>
            <p style={styles.dropText}>DCM / PNG / JPG 파일을 드래그하거나 클릭하여 업로드</p>
          </div>
        )}
      </div>

      {/* 분석 버튼 */}
      <button
        style={{
          ...styles.button,
          opacity: !file || loading ? 0.5 : 1,
          cursor: !file || loading ? "not-allowed" : "pointer",
        }}
        onClick={handleAnalyze}
        disabled={!file || loading}
      >
        {loading ? "분석 중..." : "분석하기"}
      </button>

      {/* 에러 */}
      {error && <p style={styles.error}>{error}</p>}

      {/* 결과 */}
      {result && (
        <div style={styles.resultBox}>
          {/* 판정 */}
          <div style={{ ...styles.verdict, borderColor: statusColor }}>
            <span style={{ ...styles.verdictText, color: statusColor }}>
              {result.status === "ABNORMAL" ? "⚠️ 이상 의심" : "✅ 정상"}
            </span>
            <span style={styles.prob}>이상 확률: {result.probability}%</span>
            <div style={styles.probBar}>
              <div
                style={{
                  ...styles.probFill,
                  width: `${result.probability}%`,
                  background: statusColor,
                }}
              />
            </div>
          </div>

          {/* 이미지 3열 */}
          <div style={styles.imageGrid}>
            <div style={styles.imageCard}>
              <p style={styles.imageLabel}>원본</p>
              <img
                src={`data:image/png;base64,${result.original}`}
                alt="original"
                style={styles.resultImg}
              />
            </div>
            <div style={styles.imageCard}>
              <p style={styles.imageLabel}>히트맵</p>
              <img
                src={`data:image/png;base64,${result.heatmap}`}
                alt="heatmap"
                style={styles.resultImg}
              />
            </div>
            <div style={styles.imageCard}>
              <p style={styles.imageLabel}>오버레이</p>
              <img
                src={`data:image/png;base64,${result.overlay}`}
                alt="overlay"
                style={styles.resultImg}
              />
            </div>
          </div>

          <p style={styles.disclaimer}>
            * 본 결과는 AI 보조 분석이며 의학적 진단을 대체하지 않습니다.
          </p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    background: "#0f172a",
    color: "#f1f5f9",
    fontFamily: "'Segoe UI', sans-serif",
    padding: "40px 20px",
    maxWidth: "900px",
    margin: "0 auto",
  },
  header: {
    textAlign: "center",
    marginBottom: "40px",
  },
  title: {
    fontSize: "2.5rem",
    fontWeight: "bold",
    margin: 0,
    background: "linear-gradient(90deg, #60a5fa, #a78bfa)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  },
  subtitle: {
    color: "#94a3b8",
    marginTop: "8px",
    fontSize: "1rem",
  },
  dropzone: {
    border: "2px dashed #334155",
    borderRadius: "16px",
    padding: "40px",
    textAlign: "center",
    cursor: "pointer",
    background: "#1e293b",
    transition: "border-color 0.2s",
    marginBottom: "20px",
  },
  uploadIcon: { fontSize: "3rem", marginBottom: "12px" },
  dcmIcon: { fontSize: "3rem", marginBottom: "12px" },
  dropText: { color: "#64748b", fontSize: "0.95rem" },
  previewImg: {
    maxHeight: "200px",
    borderRadius: "8px",
    marginBottom: "8px",
  },
  fileName: { color: "#94a3b8", fontSize: "0.9rem" },
  button: {
    width: "100%",
    padding: "14px",
    background: "linear-gradient(90deg, #3b82f6, #8b5cf6)",
    border: "none",
    borderRadius: "12px",
    color: "white",
    fontSize: "1.1rem",
    fontWeight: "bold",
    marginBottom: "24px",
    transition: "opacity 0.2s",
  },
  error: { color: "#ef4444", textAlign: "center", marginBottom: "16px" },
  resultBox: {
    background: "#1e293b",
    borderRadius: "16px",
    padding: "28px",
  },
  verdict: {
    border: "2px solid",
    borderRadius: "12px",
    padding: "20px",
    textAlign: "center",
    marginBottom: "24px",
  },
  verdictText: {
    fontSize: "1.8rem",
    fontWeight: "bold",
    display: "block",
    marginBottom: "8px",
  },
  prob: {
    color: "#94a3b8",
    fontSize: "0.95rem",
    display: "block",
    marginBottom: "10px",
  },
  probBar: {
    background: "#334155",
    borderRadius: "999px",
    height: "8px",
    overflow: "hidden",
  },
  probFill: {
    height: "100%",
    borderRadius: "999px",
    transition: "width 0.5s",
  },
  imageGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "16px",
    marginBottom: "16px",
  },
  imageCard: { textAlign: "center" },
  imageLabel: {
    color: "#94a3b8",
    fontSize: "0.85rem",
    marginBottom: "8px",
  },
  resultImg: {
    width: "100%",
    borderRadius: "8px",
  },
  disclaimer: {
    color: "#475569",
    fontSize: "0.8rem",
    textAlign: "center",
    marginTop: "8px",
  },
};