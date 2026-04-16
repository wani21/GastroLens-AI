import { useState } from "react";
import Dropzone from "../components/Dropzone";
import ProbabilityBars from "../components/ProbabilityBars";
import { predictImage } from "../api/client";

export default function ImagePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
      const data = await predictImage(file);
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
        <h1 className="text-4xl font-bold mb-2">Image Analysis</h1>
        <p className="text-gray-600">
          Upload a GI endoscopy image for instant classification.
        </p>
      </header>

      {!result && (
        <Dropzone
          onFileSelect={handleFile}
          accept="image/*"
          label="Drop an endoscopy image here"
          file={file}
        />
      )}

      {preview && (
        <div className="mt-6 grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Preview
            </h3>
            <img
              src={preview}
              alt="preview"
              className="w-full rounded-lg border border-gray-200"
            />
          </div>

          <div className="flex flex-col">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Result
            </h3>
            <div className="flex-1 border border-gray-200 rounded-lg p-6 bg-gray-50">
              {!result && !loading && !error && (
                <p className="text-gray-500 text-sm">
                  Click <strong>Analyze</strong> to run inference.
                </p>
              )}
              {loading && (
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  <span className="text-gray-700">Analyzing image...</span>
                </div>
              )}
              {error && (
                <div className="text-red-600 text-sm">
                  <strong>Error:</strong> {error}
                </div>
              )}
              {result && (
                <div>
                  <div className="mb-6">
                    <div className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-1">
                      Predicted Class
                    </div>
                    <div className="text-3xl font-bold capitalize text-accent">
                      {result.predicted_class}
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      Confidence: {(result.confidence * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-3">
                      All Class Probabilities
                    </div>
                    <ProbabilityBars
                      probabilities={result.probabilities}
                      predictedClass={result.predicted_class}
                    />
                  </div>
                </div>
              )}
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
