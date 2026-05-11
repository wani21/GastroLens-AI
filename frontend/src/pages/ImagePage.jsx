import { useState } from "react";
import Dropzone from "../components/Dropzone";
import ProbabilityBars from "../components/ProbabilityBars";
import { predictImage, explainImage, segmentImage, shapExplain } from "../api/client";

export default function ImagePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [gradcam, setGradcam] = useState(null);
  const [loadingGradcam, setLoadingGradcam] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);
  
  const [shap, setShap] = useState(null);
  const [loadingShap, setLoadingShap] = useState(false);
  const [showShap, setShowShap] = useState(false);
  
  const [segmentation, setSegmentation] = useState(null);
  const [loadingSegmentation, setLoadingSegmentation] = useState(false);
  const [showSegmentation, setShowSegmentation] = useState(false);

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setGradcam(null);
    setShowGradcam(false);
    setShap(null);
    setShowShap(false);
    setSegmentation(null);
    setShowSegmentation(false);
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

  const handleExplain = async () => {
    if (gradcam) {
      setShowGradcam(!showGradcam);
      return;
    }
    if (!file) return;
    setLoadingGradcam(true);
    try {
      const data = await explainImage(file);
      setGradcam(data);
      setShowGradcam(true);
    } catch (err) {
      alert("Grad-CAM Error: " + (err.response?.data?.detail || err.message));
      console.error(err);
    } finally {
      setLoadingGradcam(false);
    }
  };

  const handleShap = async () => {
    if (shap) {
      setShowShap(!showShap);
      return;
    }
    if (!file) return;
    setLoadingShap(true);
    try {
      const data = await shapExplain(file);
      setShap(data);
      setShowShap(true);
    } catch (err) {
      alert("SHAP Error: " + (err.response?.data?.detail || err.message));
      console.error(err);
    } finally {
      setLoadingShap(false);
    }
  };

  const handleSegment = async () => {
    if (segmentation) {
      setShowSegmentation(!showSegmentation);
      return;
    }
    if (!file) return;
    setLoadingSegmentation(true);
    try {
      const data = await segmentImage(file);
      setSegmentation(data);
      setShowSegmentation(true);
    } catch (err) {
      alert("U-Net Error: " + (err.response?.data?.detail || err.message));
      console.error(err);
    } finally {
      setLoadingSegmentation(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setGradcam(null);
    setShowGradcam(false);
    setShap(null);
    setShowShap(false);
    setSegmentation(null);
    setShowSegmentation(false);
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

          <div className="flex flex-col gap-6">
            <div className="transform transition-all duration-700 ease-out translate-y-0 opacity-100">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                Result
              </h3>
              <div className="border border-gray-200 rounded-lg p-6 bg-gray-50 shadow-sm hover:shadow-md transition-shadow">
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

            {/* Grad-CAM Section */}
            {result && (
              <div className="transform transition-all duration-500 ease-out">
                <button
                  onClick={handleExplain}
                  disabled={loadingGradcam}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md font-medium hover:bg-gray-50 transition-colors disabled:opacity-50"
                >
                  {loadingGradcam ? "Generating Grad-CAM..." : (showGradcam ? "Hide Grad-CAM" : "Explain with Grad-CAM")}
                </button>
                {showGradcam && gradcam && (
                  <div className="mt-4 border border-gray-200 rounded-lg p-6 bg-gray-50 animate-fade-in transition-opacity duration-300">
                    <h4 className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-3">Grad-CAM Heatmap</h4>
                    <img src={gradcam.heatmap_base64} alt="Grad-CAM" className="w-full rounded-lg border border-gray-200 mb-3" />
                    <p className="text-sm text-gray-600">The highlighted regions (red/yellow) indicate which areas of the image most influenced the model's prediction.</p>
                  </div>
                )}
              </div>
            )}

            {/* SHAP Section */}
            {result && (
              <div className="transform transition-all duration-500 ease-out">
                <button
                  onClick={handleShap}
                  disabled={loadingShap}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md font-medium hover:bg-gray-50 transition-colors disabled:opacity-50"
                >
                  {loadingShap ? "Generating SHAP Explanation..." : (showShap ? "Hide SHAP Explanation" : "SHAP Explanation")}
                </button>
                {showShap && shap && (
                  <div className="mt-4 border border-gray-200 rounded-lg p-6 bg-gray-50 animate-fade-in transition-opacity duration-300">
                    <h4 className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-3">SHAP Values</h4>
                    <img src={shap.shap_image_base64} alt="SHAP" className="w-full rounded-lg border border-gray-200 mb-3 bg-white" />
                    <p className="text-sm text-gray-600">
                      <strong>{shap.explanation}</strong><br/>
                      <span className="text-xs text-gray-500">SHAP is model-agnostic — it explains predictions by measuring how each pixel pushes the prediction toward or away from the predicted class.</span>
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* U-Net Section */}
            {result && (
              <div className="transform transition-all duration-500 ease-out">
                <button
                  onClick={handleSegment}
                  disabled={loadingSegmentation}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md font-medium hover:bg-gray-50 transition-colors disabled:opacity-50"
                >
                  {loadingSegmentation ? "Generating U-Net Mask..." : (showSegmentation ? "Hide Lesion Mask" : "Generate Lesion Mask (U-Net)")}
                </button>
                {showSegmentation && segmentation && (
                  <div className="mt-4 border border-gray-200 rounded-lg p-6 bg-gray-50 animate-fade-in transition-opacity duration-300">
                    <h4 className="text-xs uppercase text-gray-500 font-semibold tracking-wide mb-3">U-Net Segmentation</h4>
                    <div className="grid grid-cols-2 gap-4 mb-3">
                      <div>
                        <span className="text-xs text-gray-500 mb-1 block">Lesion Mask</span>
                        <img src={segmentation.mask_base64} alt="Mask" className="w-full rounded-lg border border-gray-200" />
                      </div>
                      <div>
                        <span className="text-xs text-gray-500 mb-1 block">Overlay</span>
                        <img src={segmentation.overlay_base64} alt="Overlay" className="w-full rounded-lg border border-gray-200" />
                      </div>
                    </div>
                    <p className="text-sm font-semibold text-gray-700 mb-2">Lesion coverage: {segmentation.lesion_coverage_pct}% of image area</p>
                    <p className="text-xs text-gray-600">
                      U-Net uses an encoder-decoder architecture to predict pixel-level lesion locations. The encoder extracts features (shared with the classifier), and the decoder reconstructs spatial detail using skip connections.
                    </p>
                  </div>
                )}
              </div>
            )}
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
