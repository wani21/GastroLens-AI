import { Link } from "react-router-dom";

const CLASSES = [
  {
    name: "Normal",
    description:
      "Healthy GI tract tissue — cecum, pylorus, and Z-line regions.",
  },
  {
    name: "Polyp",
    description:
      "Abnormal tissue growths that can become cancerous if left untreated.",
  },
  {
    name: "Ulcer",
    description:
      "Ulcerative colitis lesions — inflammation and open sores in the colon.",
  },
  {
    name: "Esophagitis",
    description:
      "Inflammation of the esophagus, often caused by acid reflux.",
  },
];

const METRICS = [
  { label: "Accuracy", value: "94.79%" },
  { label: "F1-Score", value: "94.68%" },
  { label: "Classes", value: "4" },
  { label: "Images Trained", value: "7,302" },
];

export default function LandingPage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      {/* Hero */}
      <section className="text-center mb-20">
        <div className="inline-block px-3 py-1 border border-gray-300 rounded-full text-xs font-medium text-gray-600 mb-6">
          Deep Learning · Transfer Learning · ResNet50
        </div>
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6">
          AI-Powered <span className="text-accent">GI Disease</span>
          <br />
          Detection
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-10">
          Upload an endoscopy image or video to get instant classification
          across four diagnostic categories — backed by a ResNet50 model
          trained on the Kvasir dataset.
        </p>
        <div className="flex gap-3 justify-center flex-wrap">
          <Link
            to="/image"
            className="px-6 py-3 bg-black text-white rounded-md font-medium hover:bg-gray-800 transition-colors"
          >
            Analyze Image →
          </Link>
          <Link
            to="/video"
            className="px-6 py-3 border border-black rounded-md font-medium hover:bg-gray-50 transition-colors"
          >
            Analyze Video
          </Link>
        </div>
      </section>

      {/* Metrics */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-20">
        {METRICS.map((m) => (
          <div
            key={m.label}
            className="border border-gray-200 rounded-lg p-6 text-center"
          >
            <div className="text-3xl font-bold text-black mb-1">{m.value}</div>
            <div className="text-sm text-gray-500">{m.label}</div>
          </div>
        ))}
      </section>

      {/* Classes */}
      <section className="mb-20">
        <h2 className="text-3xl font-bold mb-2 text-center">
          Four Diagnostic Classes
        </h2>
        <p className="text-center text-gray-600 mb-10">
          The model distinguishes between these conditions in endoscopy imagery
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          {CLASSES.map((c) => (
            <div
              key={c.name}
              className="border border-gray-200 rounded-lg p-6 hover:border-accent transition-colors"
            >
              <h3 className="font-bold text-lg mb-2 flex items-center gap-2">
                <span className="w-2 h-2 bg-accent rounded-full" />
                {c.name}
              </h3>
              <p className="text-gray-600 text-sm">{c.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="mb-10">
        <h2 className="text-3xl font-bold mb-10 text-center">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              step: "01",
              title: "Upload",
              desc: "Drop an endoscopy image or video into the browser.",
            },
            {
              step: "02",
              title: "Analyze",
              desc: "ResNet50 processes the input through a two-phase trained model.",
            },
            {
              step: "03",
              title: "Results",
              desc: "Get class prediction with confidence scores across all 4 classes.",
            },
          ].map((s) => (
            <div key={s.step} className="text-left">
              <div className="text-accent font-mono text-sm font-bold mb-2">
                {s.step}
              </div>
              <h3 className="font-bold text-xl mb-2">{s.title}</h3>
              <p className="text-gray-600">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
