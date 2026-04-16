export default function ProbabilityBars({ probabilities, predictedClass }) {
  // Sort entries by probability, descending
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-3">
      {entries.map(([cls, prob]) => {
        const isTop = cls === predictedClass;
        const pct = (prob * 100).toFixed(1);
        return (
          <div key={cls}>
            <div className="flex justify-between text-sm mb-1">
              <span
                className={`font-medium capitalize ${
                  isTop ? "text-accent" : "text-gray-700"
                }`}
              >
                {cls}
                {isTop && (
                  <span className="ml-2 text-xs bg-accent text-white px-1.5 py-0.5 rounded">
                    TOP
                  </span>
                )}
              </span>
              <span
                className={`font-mono ${
                  isTop ? "text-accent font-semibold" : "text-gray-500"
                }`}
              >
                {pct}%
              </span>
            </div>
            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ${
                  isTop ? "bg-accent" : "bg-gray-400"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
