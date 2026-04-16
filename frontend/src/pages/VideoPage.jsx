import { useState } from "react";
import Dropzone from "../components/Dropzone";
import { predictVideo } from "../api/client";

export default function VideoPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sampleRate, setSampleRate] = useState(10);

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictVideo(file, sampleRate);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <header className="mb-10">
        <h1 className="text-4xl font-bold mb-2">Video Analysis</h1>
        <p className="text-gray-600">
          Upload an endoscopy video. Frames are sampled and analyzed to find
          the dominant pathology.
        </p>
      </header>

      {!file && (
        <>
          <Dropzone
            onFileSelect={handleFile}
            accept="video/*"
            label="Drop an endoscopy video here"
            file={file}
          />
          <div className="mt-4 flex items-center gap-3">
            <label className="text-sm text-gray-600">Sample every</label>
            <input
              type="number"
              min="1"
              max="100"
              value={sampleRate}
              onChange={(e) => setSampleRate(Number(e.target.value))}
              className="w-20 px-3 py-1.5 border border-gray-300 rounded-md text-sm"
            />
            <span className="text-sm text-gray-600">frames</span>
          </div>
        </>
      )}

      {preview && (
        <div className="mt-6 grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Preview
            </h3>
            <video
              src={preview}
              controls
              className="w-full rounded-lg border border-gray-200"
            />
          </div>

          <div className="flex flex-col">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Result
            </h3>
            <div className="flex-1 border border-gray-200 rounded-lg p-6 bg-gray-50 overflow-y-auto max-h-[500px]">
              {!result && !loading && !error && (
                <p className="text-gray-500 text-sm">
                  Click <strong>Analyze</strong> to sample and classify frames.
                </p>
              )}
              {loading && (
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  <span className="text-gray-700">
                    Analyzing video... (may take a minute)
                  </span>
                </div>
              )}
              {error && (
                <div className="text-red-600 text-sm">
                  <strong>Error:</strong> {error}
                </div>
              )}
              {result && <VideoResult result={result} />}
            </div>
          </div>
        </div>
      )}

      {file && (
        <div className="mt-8 flex gap-3 justify-center">
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="px-6 py-3 bg-black text-white rounded-md font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          <button
            onClick={reset}
            className="px-6 py-3 border border-gray-300 rounded-md font-medium hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
        </div>
      )}
    </div>
  );
}

function VideoResult({ result }) {
  const totalSampled = result.sampled_frames;
  const entries = Object.entries(result.class_distribution).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div className="space-y-6">
      <div>
        <div className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-1">
          Dominant Class
        </div>
        <div className="text-3xl font-bold capitalize text-accent">
          {result.dominant_class}
        </div>
        <div className="text-sm text-gray-600 mt-1">
          Found in {(result.dominant_class_ratio * 100).toFixed(1)}% of sampled
          frames
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <Stat label="Total Frames" value={result.total_frames} />
        <Stat label="Sampled" value={result.sampled_frames} />
        <Stat label="FPS" value={result.fps} />
      </div>

      <div>
        <div className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-3">
          Class Distribution
        </div>
        <div className="space-y-2">
          {entries.map(([cls, count]) => {
            const pct = ((count / totalSampled) * 100).toFixed(1);
            const isTop = cls === result.dominant_class;
            return (
              <div key={cls}>
                <div className="flex justify-between text-sm mb-1">
                  <span
                    className={`capitalize ${
                      isTop ? "text-accent font-semibold" : "text-gray-700"
                    }`}
                  >
                    {cls}
                  </span>
                  <span className="font-mono text-gray-500">
                    {count}/{totalSampled} ({pct}%)
                  </span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      isTop ? "bg-accent" : "bg-gray-400"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer font-semibold text-gray-700 hover:text-black">
          Per-Frame Results ({result.frame_results.length})
        </summary>
        <div className="mt-3 space-y-1 font-mono text-xs max-h-60 overflow-y-auto">
          {result.frame_results.map((r) => (
            <div key={r.frame_index} className="flex justify-between">
              <span className="text-gray-500">Frame #{r.frame_index}</span>
              <span className="capitalize text-gray-800">
                {r.predicted_class}
              </span>
              <span className="text-gray-500">
                {(r.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="border border-gray-200 rounded-md p-3 bg-white">
      <div className="text-xs text-gray-500 uppercase">{label}</div>
      <div className="text-lg font-bold">{value}</div>
    </div>
  );
}
